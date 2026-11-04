import pkg_resources
import pytest
import torch

from sam3.model.text_encoder_ve import TextTransformer, VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.train import model_init

BPE = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")


def _text_encoder(ctx, pad_to_longest=False, width=64, layers=2):
    return VETextEncoder(
        d_model=32,
        tokenizer=SimpleTokenizer(bpe_path=BPE),
        width=width,
        heads=4,
        layers=layers,
        context_length=ctx,
        use_act_checkpoint=False,
        pad_to_longest=pad_to_longest,
    ).eval()


def test_text_context_extension_keeps_short_prompts_identical():
    torch.manual_seed(0)
    short = _text_encoder(32)
    for p in short.parameters():
        torch.nn.init.normal_(p, std=0.05)
    long = _text_encoder(256)
    long.load_state_dict(short.state_dict())
    pos = long.encoder.positional_embedding
    torch.testing.assert_close(pos[:32], short.encoder.positional_embedding)
    torch.testing.assert_close(
        pos[32:], short.encoder.positional_embedding[-1].expand(224, -1)
    )

    texts = ["a dog", "the second cyclist from the left in a yellow jersey"]
    with torch.no_grad():
        m_s, f_s, _ = short(texts, device="cpu")
        m_l, f_l, _ = long(texts, device="cpu")
    valid = ~m_s  # True on real tokens; [B, 32]
    torch.testing.assert_close(m_l[:, :32], m_s)
    assert m_l[:, 32:].all()  # nothing but padding past 32 for these prompts
    # Features on real tokens are identical (causal encoder, same positions).
    torch.testing.assert_close(
        f_l[:32].transpose(0, 1)[valid], f_s.transpose(0, 1)[valid]
    )


def test_long_prompt_no_longer_truncated():
    enc = _text_encoder(256)
    text = " ".join(["word"] * 100)
    with torch.no_grad():
        mask, _, _ = enc([text], device="cpu")
    assert int((~mask).sum()) == 102  # sot + 100 + eot
    enc32 = _text_encoder(32)
    with torch.no_grad():
        mask32, _, _ = enc32([text], device="cpu")
    assert int((~mask32).sum()) == 32


def test_pad_to_longest_same_features_on_real_tokens():
    torch.manual_seed(0)
    full = _text_encoder(256)
    for p in full.parameters():
        torch.nn.init.normal_(p, std=0.05)
    trimmed = _text_encoder(256, pad_to_longest=True)
    trimmed.load_state_dict(full.state_dict())
    texts = ["a dog", "a red car next to the tree"]
    with torch.no_grad():
        m_f, f_f, e_f = full(texts, device="cpu")
        m_t, f_t, e_t = trimmed(texts, device="cpu")
    L = m_t.shape[1]
    assert L == int((~m_f).sum(1).max()) < 256
    torch.testing.assert_close(m_t, m_f[:, :L])
    torch.testing.assert_close(f_t, f_f[:L])


# ---------------------------------------------------------------------------
# PE -> SAM3 mapping
# ---------------------------------------------------------------------------


def _fake_pe_state_dict(model, depth):
    """A PE CLIP state dict with the shapes of the SAM3 encoders, random values."""
    sd = model.state_dict()
    v, t = model_init.VISION_PREFIX, model_init.TEXT_PREFIX
    pe = {
        "visual.conv1.weight": torch.randn_like(sd[v + "patch_embed.proj.weight"]),
        "visual.class_embedding": torch.randn(1024),
        "visual.positional_embedding": torch.randn(577, 1024),
        "visual.ln_pre.weight": torch.randn(1024),
        "visual.ln_pre.bias": torch.randn(1024),
        "visual.ln_post.weight": torch.randn(1024),
        "visual.ln_post.bias": torch.randn(1024),
        "visual.proj": torch.randn(1024, 1024),
        "visual.attn_pool.probe": torch.randn(1, 1, 1024),
        "text_projection": torch.randn(1024, 1024),
        "logit_scale": torch.tensor(1.0),
    }
    pe_names = {
        "norm1": "ln_1",
        "norm2": "ln_2",
        "attn.qkv.weight": "attn.in_proj_weight",
        "attn.qkv.bias": "attn.in_proj_bias",
        "attn.proj": "attn.out_proj",
        "mlp.fc1": "mlp.c_fc",
        "mlp.fc2": "mlp.c_proj",
    }
    for i in range(depth):
        for sam, pen in pe_names.items():
            for suffix in ("", ".weight", ".bias"):
                key = f"{v}blocks.{i}.{sam}{suffix}"
                if key in sd and not key.endswith("freqs_cis"):
                    pe[f"visual.transformer.resblocks.{i}.{pen}{suffix}"] = (
                        torch.randn_like(sd[key])
                    )
    for key, val in sd.items():
        if key.startswith(t) and not key.endswith("text_projection"):
            sub = key[len(t) :]
            if sub == "positional_embedding":
                pe[sub] = torch.randn(32, val.shape[1])
            else:
                pe[sub] = torch.randn_like(val)
    return pe


@pytest.fixture(scope="module")
def small_model():
    return model_init.build_sam3_image_model_for_training(
        vit_depth=2,
        vit_global_att_blocks=(1,),
        pe_checkpoint=None,
        text_context_length=256,
    )


def test_pe_mapping_loads_every_encoder_tensor(small_model, tmp_path):
    pe = _fake_pe_state_dict(small_model, depth=2)
    path = tmp_path / "pe.pt"
    torch.save(pe, path)
    model_init.load_pe_into_sam3(small_model, str(path))
    sd = small_model.state_dict()
    v, t = model_init.VISION_PREFIX, model_init.TEXT_PREFIX
    torch.testing.assert_close(
        sd[v + "blocks.1.attn.qkv.weight"],
        pe["visual.transformer.resblocks.1.attn.in_proj_weight"],
    )
    torch.testing.assert_close(
        sd[v + "blocks.0.mlp.fc2.bias"],
        pe["visual.transformer.resblocks.0.mlp.c_proj.bias"],
    )
    torch.testing.assert_close(
        sd[v + "pos_embed"][0], pe["visual.positional_embedding"]
    )
    torch.testing.assert_close(
        sd[v + "patch_embed.proj.weight"], pe["visual.conv1.weight"]
    )
    torch.testing.assert_close(
        sd[t + "token_embedding.weight"], pe["token_embedding.weight"]
    )
    torch.testing.assert_close(
        sd[t + "positional_embedding"][:32], pe["positional_embedding"]
    )
    torch.testing.assert_close(
        sd[t + "positional_embedding"][200], pe["positional_embedding"][-1]
    )


def test_pe_mapping_rejects_depth_mismatch(small_model, tmp_path):
    pe = _fake_pe_state_dict(small_model, depth=2)
    pe["visual.transformer.resblocks.2.ln_1.weight"] = torch.randn(1024)  # a 3rd block
    path = tmp_path / "pe.pt"
    torch.save(pe, path)
    with pytest.raises(ValueError, match="missing in SAM3"):
        model_init.load_pe_into_sam3(small_model, str(path))
    del pe["visual.transformer.resblocks.2.ln_1.weight"]
    del pe["visual.transformer.resblocks.1.mlp.c_fc.weight"]  # a block too few
    torch.save(pe, path)
    with pytest.raises(ValueError, match="trunk key not in PE"):
        model_init.load_pe_into_sam3(small_model, str(path))


def test_default_global_blocks():
    assert model_init.default_global_att_blocks(32) == (7, 15, 23, 31)
    assert model_init.default_global_att_blocks(24) == (5, 11, 17, 23)


# ---------------------------------------------------------------------------
# Batched matching over decoder layers
# ---------------------------------------------------------------------------


def test_batched_matching_equals_per_layer(small_model):
    from sam3.train.matcher import BinaryHungarianMatcherV2

    torch.manual_seed(0)
    small_model.matcher = BinaryHungarianMatcherV2(
        focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0
    )
    B, Q, L = 5, 20, 6
    num_boxes = torch.tensor([3, 0, 7, 1, 25])  # one with more targets than queries
    boxes = torch.rand(int(num_boxes.sum()), 4) * 0.4 + 0.2
    padded = torch.zeros(B, int(num_boxes.max()), 4)
    i = 0
    for b, n in enumerate(num_boxes.tolist()):
        padded[b, :n] = boxes[i : i + n]
        i += n
    targets = {"boxes": boxes, "boxes_padded": padded, "num_boxes": num_boxes}

    def outs():
        torch.manual_seed(1)
        layers = [
            {
                "pred_logits": torch.randn(B, Q, 1),
                "pred_boxes": torch.rand(B, Q, 4) * 0.5 + 0.25,
            }
            for _ in range(L)
        ]
        out = dict(layers[-1])
        out["aux_outputs"] = layers[:-1]
        return out

    ref = outs()
    small_model.batch_matching_over_layers = False
    small_model._compute_matching(ref, targets)
    got = outs()
    small_model.batch_matching_over_layers = True
    small_model._compute_matching(got, targets)
    for r, g in zip([ref] + ref["aux_outputs"], [got] + got["aux_outputs"]):
        for a, b in zip(r["indices"], g["indices"]):
            if a is None:
                assert b is None
            else:
                torch.testing.assert_close(a, b)
