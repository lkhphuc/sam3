"""Masks for matched queries only (DeferredMasks): same values, losses and
gradients as the dense pred_masks path."""

import random

import pytest
import torch

from sam3.model.maskformer_segmentation import DeferredMasks
from sam3.train.loss.loss_fns import Masks


def _deferred(B=4, Q=10, C=8, H=12, W=9, shared=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, Q, C, generator=g, requires_grad=True)
    inst = torch.randn(
        *(() if shared else (B,)), C, H, W, generator=g, requires_grad=True
    )
    return DeferredMasks(q, inst)


@pytest.mark.parametrize("shared", [False, True])
def test_gather_equals_dense_rows_and_grads(shared):
    d = _deferred(shared=shared)
    b = torch.tensor([3, 0, 3, 1, 0, 3])
    qi = torch.tensor([2, 9, 5, 0, 1, 2])  # repeated pair (3, 2): o2m style
    w = torch.randn(len(b), 12, 9)
    got = d.gather(b, qi)
    ref = d.materialize()[b, qi]
    torch.testing.assert_close(got, ref)
    g1 = torch.autograd.grad((got * w).sum(), [d.mask_queries, d.instance_embeds])
    g2 = torch.autograd.grad((ref * w).sum(), [d.mask_queries, d.instance_embeds])
    for a, c in zip(g1, g2):
        torch.testing.assert_close(a, c)
    assert d.gather(b[:0], qi[:0]).shape == (0, 12, 9)


def test_slice_queries():
    d = _deferred()
    torch.testing.assert_close(
        d.slice_queries(None, 4).materialize(), d.materialize()[:, :4]
    )
    torch.testing.assert_close(
        d.slice_queries(4, None).materialize(), d.materialize()[:, 4:]
    )


@pytest.mark.parametrize(
    "kw",
    [
        dict(
            num_sample_points=50,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            target_sampling="nearest",
        ),
        dict(
            num_sample_points=50,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            target_sampling="bilinear",
        ),
        dict(),  # full-resolution loss
    ],
)
def test_masks_loss_deferred_equals_dense(kw):
    d = _deferred(B=3, Q=6, H=12, W=12)
    loss = Masks(weight_dict={"loss_mask": 1.0, "loss_dice": 1.0}, **kw)
    n = 5
    targets = {
        "masks": torch.rand(n, 48, 48) > 0.5,
        "is_valid_mask": torch.tensor([True, True, False, True, True]),
    }
    indices = (
        torch.tensor([0, 0, 2, 1, 2]),
        torch.tensor([1, 4, 0, 5, 3]),
        torch.tensor([2, 0, 1, 4, 3]),
    )

    def run(pred):
        torch.manual_seed(0)
        out = loss.get_loss({"pred_masks": pred}, targets, indices, num_boxes=4.0)
        total = out["loss_mask"] + out["loss_dice"]
        grads = torch.autograd.grad(total, [d.mask_queries, d.instance_embeds])
        return out, grads

    ref, g_ref = run(d.materialize())
    got, g_got = run(d)
    for k in ("loss_mask", "loss_dice"):
        torch.testing.assert_close(got[k], ref[k])
    for a, c in zip(g_got, g_ref):
        torch.testing.assert_close(a, c)


@pytest.mark.timeout(1800)
def test_full_model_loss_equal_with_and_without_flag():
    """Real (tiny) model + real loss config: identical losses with the flag on."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from hydra.utils import instantiate
    from test_streaming_trainer import _cfg, FakeLoader

    import tempfile

    import torch.distributed as dist

    if not dist.is_initialized():  # Sam3LossWrapper normalises over ranks
        dist.init_process_group(
            "gloo", init_method=f"file://{tempfile.mkdtemp()}/pg", rank=0, world_size=1
        )
    cfg = _cfg(tempfile.mkdtemp(), 1)
    model = instantiate(cfg.trainer.model, _convert_="all")
    model.batch_matching_over_layers = True
    loss_fn = instantiate(cfg.trainer.loss, _convert_="all")
    batch = next(b for b in FakeLoader().batches if b.boxes.shape[0] > 0)
    bd = batch.to_batched_datapoint(torch.device("cpu"))

    def run(flag):
        model.masks_for_matched_queries_only = flag
        random.seed(0)
        torch.manual_seed(0)
        out = model(bd)
        if flag:
            (stage_out,) = out.output[0]  # one find stage, one step
            assert isinstance(stage_out["pred_masks"], DeferredMasks)
            assert isinstance(stage_out["pred_masks_o2m"], DeferredMasks)
        losses = loss_fn(out, [model.back_convert(x) for x in bd.find_targets])
        model.zero_grad(set_to_none=True)
        losses["core_loss"].backward()  # through the act-checkpointed seg head
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if n in watched}
        vals = {k: float(v) for k, v in losses.items() if isinstance(v, torch.Tensor)}
        del out, losses
        return vals, grads

    watched = {
        "segmentation_head.instance_seg_head.weight",
        "segmentation_head.mask_predictor.mask_embed.layers.0.weight",
        "transformer.decoder.layers.5.cross_attn.in_proj_weight",
    }
    assert watched <= {n for n, _ in model.named_parameters()}
    ref, g_ref = run(False)
    got, g_got = run(True)
    assert set(ref) == set(got)
    for k in ref:
        assert got[k] == pytest.approx(ref[k], rel=1e-4, abs=1e-6), k
    assert ref["loss_mask"] > 0
    for n in sorted(watched):
        # fp32 rounding only (the per-prompt einsums group sums differently):
        # measured ~1e-6 relative in norm.
        rel = (g_got[n] - g_ref[n]).norm() / g_ref[n].norm()
        assert g_ref[n].norm() > 0 and rel < 1e-5, (n, float(rel))
