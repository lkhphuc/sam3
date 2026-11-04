# pyre-unsafe
"""Build a SAM3 image model for training from scratch, initialised from
Perception Encoder (PE) vision and text towers, the way SAM3 itself was trained.

Only the two encoders come from PE; the neck, fusion encoder, decoder, scoring
and segmentation heads start from their random init.

Vision tower. SAM3's released trunk (width 1024, 32 blocks, MLP ratio 4.625,
pretrain size 336) does not match any public PE checkpoint. The closest public
one is PE-Core-L14-336: width 1024, 24 blocks, MLP ratio 4.0, the same
336-pixel pretraining with a CLS position, abs pos-emb and 2D RoPE. So the
trunk shape is configurable, and the defaults below follow PE-Core-L14-336.
Global-attention blocks are placed like SAM3 places them (the last block of
every quarter); the other blocks use 24x24 windows, the PE pretraining grid.

Text tower. SAM3's text encoder is the PE-Core-L14-336 text tower (width 1024,
24 layers, 16 heads, context 32). Its position table is extended to
`text_context_length` on load (TextTransformer._load_from_state_dict).
"""

import logging
import re
from typing import Dict, Optional, Sequence, Tuple

import torch

from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT

logger = logging.getLogger(__name__)

VISION_PREFIX = "backbone.vision_backbone.trunk."
TEXT_PREFIX = "backbone.language_backbone.encoder."

# PE-Core-L14-336 on the HF hub (core/vision_encoder/config.py in
# facebookresearch/perception_models): repo facebook/<name>, file <name>.pt.
DEFAULT_PE_CHECKPOINT = "hf://facebook/PE-Core-L14-336:PE-Core-L14-336.pt"


def default_global_att_blocks(depth: int, num_global: int = 4) -> Tuple[int, ...]:
    """The last block of each of `num_global` equal stages: (7, 15, 23, 31) for 32."""
    if depth % num_global:
        raise ValueError(f"depth {depth} is not divisible by {num_global}")
    step = depth // num_global
    return tuple(step * (i + 1) - 1 for i in range(num_global))


def build_sam3_image_model_for_training(
    *,
    bpe_path: Optional[str] = None,
    vit_depth: int = 24,
    vit_mlp_ratio: float = 4.0,
    vit_global_att_blocks: Optional[Sequence[int]] = None,
    vit_drop_path_rate: float = 0.1,
    text_context_length: int = 256,
    text_pad_to_longest: bool = True,
    enable_segmentation: bool = True,
    compile_vision: bool = False,
    use_fa3: bool = False,
    pe_checkpoint: Optional[str] = DEFAULT_PE_CHECKPOINT,
    sam3_checkpoint: Optional[str] = None,
):
    """A trainable Sam3Image (train mode, CPU), initialised from PE.

    pe_checkpoint: a PE CLIP checkpoint (local path or "hf://repo:file") whose
        vision and text towers initialise the trunk and the text encoder; None
        leaves them random.
    sam3_checkpoint: alternatively, a full SAM3 checkpoint (only for a trunk
        built with SAM3's own shape: depth 32, MLP ratio 4.625); applied
        instead of pe_checkpoint.
    """
    import pkg_resources
    from sam3 import model_builder as mb

    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )
    compile_mode = "default" if compile_vision else None
    if vit_global_att_blocks is None:
        vit_global_att_blocks = default_global_att_blocks(vit_depth)
    trunk = ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=vit_depth,
        num_heads=16,
        mlp_ratio=vit_mlp_ratio,
        norm_layer="LayerNorm",
        drop_path_rate=vit_drop_path_rate,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=tuple(vit_global_att_blocks),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
        use_fa3=use_fa3,
    )
    position_encoding = mb._create_position_encoding(precompute_resolution=1008)
    neck = mb._create_vit_neck(
        position_encoding, trunk, enable_inst_interactivity=False
    )
    text_encoder = VETextEncoder(
        tokenizer=SimpleTokenizer(bpe_path=bpe_path),
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
        context_length=text_context_length,
        pad_to_longest=text_pad_to_longest,
    )
    backbone = mb._create_vl_backbone(neck, text_encoder)
    transformer = mb._create_sam3_transformer(use_fa3=use_fa3)
    dot_prod_scoring = mb._create_dot_product_scoring()
    segmentation_head = (
        mb._create_segmentation_head(compile_mode=compile_mode, use_fa3=use_fa3)
        if enable_segmentation
        else None
    )
    geometry_encoder = mb._create_geometry_encoder()
    model = mb._create_sam3_model(
        backbone,
        transformer,
        geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        None,
        eval_mode=False,
    )
    if sam3_checkpoint is not None:
        mb._load_checkpoint(model, sam3_checkpoint)
    elif pe_checkpoint is not None:
        load_pe_into_sam3(model, pe_checkpoint)
    model.train()
    return model


# ---------------------------------------------------------------------------
# PE -> SAM3 key mapping
# ---------------------------------------------------------------------------

_VISION_RULES = [
    (r"conv1\.weight", "patch_embed.proj.weight"),
    (r"positional_embedding", "pos_embed"),
    (r"ln_pre\.(weight|bias)", r"ln_pre.\1"),
    (r"transformer\.resblocks\.(\d+)\.ln_1\.(weight|bias)", r"blocks.\1.norm1.\2"),
    (r"transformer\.resblocks\.(\d+)\.ln_2\.(weight|bias)", r"blocks.\1.norm2.\2"),
    (
        r"transformer\.resblocks\.(\d+)\.attn\.in_proj_weight",
        r"blocks.\1.attn.qkv.weight",
    ),
    (r"transformer\.resblocks\.(\d+)\.attn\.in_proj_bias", r"blocks.\1.attn.qkv.bias"),
    (
        r"transformer\.resblocks\.(\d+)\.attn\.out_proj\.(weight|bias)",
        r"blocks.\1.attn.proj.\2",
    ),
    (
        r"transformer\.resblocks\.(\d+)\.mlp\.c_fc\.(weight|bias)",
        r"blocks.\1.mlp.fc1.\2",
    ),
    (
        r"transformer\.resblocks\.(\d+)\.mlp\.c_proj\.(weight|bias)",
        r"blocks.\1.mlp.fc2.\2",
    ),
    (r"transformer\.resblocks\.(\d+)\.ls_1\.gamma", r"blocks.\1.ls1.gamma"),
    (r"transformer\.resblocks\.(\d+)\.ls_2\.gamma", r"blocks.\1.ls2.gamma"),
]
# PE vision keys with no SAM3 counterpart: the CLS token is dropped
# (retain_cls_token=False), SAM3 has no ln_post / projection / attention pooling.
_VISION_IGNORED = re.compile(r"^(class_embedding|ln_post\..*|proj|attn_pool\..*)$")

_TEXT_KEYS = re.compile(
    r"^(token_embedding\.weight|positional_embedding|ln_final\.(weight|bias)|transformer\.resblocks\..*)$"
)
# SAM3 uses the per-token text features (resized to d_model), not the pooled,
# projected embedding: text_projection is never used, and its shape differs.
_TEXT_IGNORED = re.compile(r"^(text_projection|logit_scale)$")

# SAM3 parameters/buffers that PE does not provide.
_TRUNK_NOT_FROM_PE = re.compile(
    r"^blocks\.\d+\.attn\.freqs_cis$"
)  # RoPE tables, computed
_TEXT_NOT_FROM_PE = re.compile(r"^text_projection$")


def _load_pe_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        repo, filename = path[len("hf://") :].split(":")
        path = hf_hub_download(repo_id=repo, filename=filename)
    from iopath.common.file_io import g_pathmgr

    with g_pathmgr.open(path, "rb") as f:
        sd = torch.load(f, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    return {
        k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()
    }


def map_pe_state_dict(
    pe_sd: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], list]:
    """PE CLIP state dict -> SAM3 keys. Returns (mapped, unmapped PE keys)."""
    mapped, unmapped = {}, []
    for key, value in pe_sd.items():
        if key.startswith("visual."):
            sub = key[len("visual.") :]
            if _VISION_IGNORED.match(sub):
                continue
            for pattern, repl in _VISION_RULES:
                if re.fullmatch(pattern, sub):
                    target = re.sub(pattern, repl, sub)
                    if target == "pos_embed":
                        value = value.unsqueeze(0) if value.dim() == 2 else value
                    mapped[VISION_PREFIX + target] = value
                    break
            else:
                unmapped.append(key)
        elif _TEXT_IGNORED.match(key):
            continue
        elif _TEXT_KEYS.match(key):
            mapped[TEXT_PREFIX + key] = value
        else:
            unmapped.append(key)
    return mapped, unmapped


def load_pe_into_sam3(model: torch.nn.Module, pe_checkpoint: str) -> None:
    """Copy the PE vision and text towers into the SAM3 trunk and text encoder.

    Strict about everything that matters: every PE tower key must map to a
    SAM3 key of the same shape (except the text position table, which may be
    shorter and is extended), and every trunk / text-encoder parameter must be
    covered. Raises ValueError otherwise.
    """
    mapped, unmapped = map_pe_state_dict(_load_pe_state_dict(pe_checkpoint))
    if unmapped:
        raise ValueError(f"PE keys with no SAM3 counterpart: {sorted(unmapped)[:20]}")
    own = model.state_dict()
    problems = []
    for key, value in mapped.items():
        if key not in own:
            problems.append(f"missing in SAM3: {key}")
        elif own[key].shape != value.shape and not (
            key == TEXT_PREFIX + "positional_embedding"
            and value.shape[0] <= own[key].shape[0]
            and value.shape[1:] == own[key].shape[1:]
        ):
            problems.append(
                f"shape {key}: SAM3 {tuple(own[key].shape)} vs PE {tuple(value.shape)}"
            )
    for key in own:
        if key.startswith(VISION_PREFIX):
            sub = key[len(VISION_PREFIX) :]
            if key not in mapped and not _TRUNK_NOT_FROM_PE.match(sub):
                problems.append(f"trunk key not in PE: {key}")
        elif key.startswith(TEXT_PREFIX):
            sub = key[len(TEXT_PREFIX) :]
            if key not in mapped and not _TEXT_NOT_FROM_PE.match(sub):
                problems.append(f"text key not in PE: {key}")
    if problems:
        raise ValueError(
            "PE checkpoint does not match the SAM3 encoders (check vit_depth / "
            "vit_mlp_ratio against the PE config):\n  " + "\n  ".join(problems[:30])
        )
    # The text TextTransformer extends a shorter position table in its own
    # _load_from_state_dict, so load module by module.
    trunk = model.backbone.vision_backbone.trunk
    text = model.backbone.language_backbone.encoder
    trunk_sd = {
        k[len(VISION_PREFIX) :]: v
        for k, v in mapped.items()
        if k.startswith(VISION_PREFIX)
    }
    text_sd = {
        k[len(TEXT_PREFIX) :]: v for k, v in mapped.items() if k.startswith(TEXT_PREFIX)
    }
    missing, unexpected = trunk.load_state_dict(trunk_sd, strict=False)
    assert not unexpected and all(_TRUNK_NOT_FROM_PE.match(k) for k in missing), (
        missing,
        unexpected,
    )
    missing, unexpected = text.load_state_dict(text_sd, strict=False)
    assert not unexpected and all(_TEXT_NOT_FROM_PE.match(k) for k in missing), (
        missing,
        unexpected,
    )
    logger.info(
        f"Initialised the SAM3 trunk ({len(trunk_sd)} tensors) and text encoder "
        f"({len(text_sd)} tensors) from {pe_checkpoint}"
    )


# Activation-checkpointing switches that can be turned off. The fusion encoder
# and the decoder assert that checkpointing is on in training
# (model/encoder.py, model/decoder.py), so they are not switchable here.
ACT_CKPT_PARTS = ("vit", "text", "seg_head", "geometry")


def set_activation_checkpointing(
    model: torch.nn.Module, off: Sequence[str] = ()
) -> None:
    """Turn activation checkpointing off for the named parts, on for the others."""
    m = model.module if hasattr(model, "module") else model
    unknown = set(off) - set(ACT_CKPT_PARTS)
    if unknown:
        raise ValueError(
            f"unknown act-ckpt parts {sorted(unknown)}; choose from {ACT_CKPT_PARTS}"
        )
    m.backbone.vision_backbone.trunk.use_act_checkpoint = "vit" not in off
    m.backbone.language_backbone.encoder.transformer.grad_checkpointing = (
        "text" not in off
    )
    m.use_act_checkpoint_seg_head = "seg_head" not in off
    if m.segmentation_head is not None:
        m.segmentation_head.act_ckpt = "seg_head" not in off
    m.geometry_encoder.use_act_ckpt = "geometry" not in off
