"""Check that the PE -> SAM3 encoder init transfers PE's features (run on a GPU box).

Needs facebookresearch/perception_models installed (`core.vision_encoder.pe`).

    python scripts/falcon_v15/check_pe_init.py --pe PE-Core-L14-336 --image some.jpg

Vision: SAM3's trunk drops the CLS token from the sequence and uses windowed
attention in most blocks, so it cannot match PE bit for bit. The check builds
the SAM3 trunk with global attention in every block at PE's own 336 px input,
loads the PE weights through sam3.train.model_init, and compares its patch
features to PE's (cosine similarity per patch). It also tries the other RoPE
axis layout (x/y halves swapped): the mapping is right if "as mapped" is
clearly higher than "rope axes swapped". If not, the RoPE layouts differ and
the trunk's freqs_cis must be permuted before training.

Text: the SAM3 text encoder (context extended to 256) and PE's text tower must
give the same per-token features on prompts of <= 32 tokens (max abs diff ~1e-5
in fp32).
"""

import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.train import model_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", default="PE-Core-L14-336")
    ap.add_argument("--image", required=True)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    args = ap.parse_args()

    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as pe_transforms

    dev = "cuda"
    clip = pe.CLIP.from_config(args.pe, pretrained=True).to(dev).eval()
    sd = {k: v for k, v in clip.state_dict().items()}
    mapped, unmapped = model_init.map_pe_state_dict(sd)
    assert not unmapped, unmapped

    # --- vision -----------------------------------------------------------
    trunk = ViT(
        img_size=336,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=args.depth,
        num_heads=16,
        mlp_ratio=args.mlp_ratio,
        norm_layer="LayerNorm",
        drop_path_rate=0.0,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=tuple(range(args.depth)),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=0,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        use_act_checkpoint=False,
    )
    vp = model_init.VISION_PREFIX
    missing, unexpected = trunk.load_state_dict(
        {k[len(vp) :]: v for k, v in mapped.items() if k.startswith(vp)}, strict=False
    )
    assert not unexpected and all(k.endswith("freqs_cis") for k in missing), (
        missing,
        unexpected,
    )
    trunk = trunk.to(dev).eval()

    preprocess = pe_transforms.get_image_transform(clip.image_size)
    img = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = clip.visual.forward_features(
            img, norm=False, strip_cls_token=True
        )  # [1, 576, 1024]
        got = trunk(img)[-1].flatten(2).transpose(1, 2)  # [1, 576, 1024]
        cos = F.cosine_similarity(ref.float(), got.float(), dim=-1)
        print(
            f"vision, as mapped:        patch cosine mean {cos.mean():.4f} min {cos.min():.4f}"
        )
        for blk in trunk.blocks:
            f = blk.attn.freqs_cis
            half = f.shape[-1] // 2
            blk.attn.freqs_cis = torch.cat([f[..., half:], f[..., :half]], dim=-1)
        got_sw = trunk(img)[-1].flatten(2).transpose(1, 2)
        cos_sw = F.cosine_similarity(ref.float(), got_sw.float(), dim=-1)
        print(
            f"vision, rope axes swapped: patch cosine mean {cos_sw.mean():.4f} min {cos_sw.min():.4f}"
        )
    print("(SAM3 drops the CLS token, so expect high but not 1.0 for the right layout)")

    # --- text -------------------------------------------------------------
    import pkg_resources

    bpe = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    text = VETextEncoder(
        d_model=256,
        tokenizer=SimpleTokenizer(bpe_path=bpe),
        width=1024,
        heads=16,
        layers=24,
        context_length=256,
        use_act_checkpoint=False,
    )
    tp = model_init.TEXT_PREFIX
    text.encoder.load_state_dict(
        {k[len(tp) :]: v for k, v in mapped.items() if k.startswith(tp)}, strict=False
    )
    text = text.to(dev).eval()
    prompts = ["a dog", "the insulator at the lower crossarm tip over the road"]
    toks = text.tokenizer(prompts, context_length=32).to(dev)
    with torch.no_grad():
        _, ref_tokens = (
            clip.encode_text(toks, return_tokens=True)
            if hasattr(clip, "encode_text")
            else (None, None)
        )
        _, sam_tokens = text.encoder(toks)
        sam_long = text.encoder(text.tokenizer(prompts, context_length=256).to(dev))[1][
            :, :32
        ]
    valid = toks != 0
    print(
        "text, 32 vs 256 context, max abs diff on real tokens:",
        (sam_tokens[valid] - sam_long[valid]).abs().max().item(),
    )
    if ref_tokens is not None:
        print(
            "text, PE vs SAM3 per-token max abs diff:",
            (ref_tokens[valid] - sam_tokens[valid]).abs().max().item(),
        )
    else:
        print(
            "text: compare PE's per-token features manually (no return_tokens API found)"
        )


if __name__ == "__main__":
    main()
