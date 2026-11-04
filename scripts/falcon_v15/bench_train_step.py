"""SAM3 training-step benchmark on synthetic batches (GPU).

Builds the model and loss exactly as `configs/falcon_v15/sam3_falcon_v15_seg.yaml`
does (PE weights are not needed for timing), then times full training steps
(H2D copy + mask unpack, forward, loss, backward, clip, AdamW) over a grid of
per-GPU batch shapes and options. One synthetic batch per configuration is
reused for every step, so the data loader is not part of the measurement (see
inspect_loader.py for that).

Modes
  sweep     (default) time every combination of the grid; JSONL + table.
  numerics  one batch, forward + loss only, compare loss terms across the
            option flags (batch matching, matched-only masks, target sampling,
            bf16 vs fp32).

Single GPU:
  python scripts/falcon_v15/bench_train_step.py --queries 128,256,512 --images 8,16 \
      --matched-masks 0,1 --out /tmp/bench.jsonl
Under torchrun (DDP; with TORCH_DISTRIBUTED_DEBUG=DETAIL this also names any
parameter that gets no gradient when find_unused_parameters is false):
  TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node 8 \
      scripts/falcon_v15/bench_train_step.py --queries 256 --images 16 --steps 5
Numerics:
  python scripts/falcon_v15/bench_train_step.py --mode numerics --queries 32 --images 4

See docs/FALCON_V15_GPU_RUNBOOK.md for what to run and what to look for.
"""

import argparse
import contextlib
import itertools
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist

from sam3.train.data.falcon_energon import collate_samples, Sam3FalconSample
from sam3.train.data.falcon_schema import pack_masks

CONFIG = "configs/falcon_v15/sam3_falcon_v15_seg.yaml"
WORDS = (
    "red blue small large wooden metal rusty broken left right top bottom car dog "
    "person tree sign insulator tower cable window door chair table bottle cup bird "
    "boat bicycle road building roof wall fence pole light lamp grape leaf"
).split()


# ---------------------------------------------------------------------------
# Synthetic batches
# ---------------------------------------------------------------------------


def synth_batch(
    *,
    num_images: int,
    num_queries: int,
    instances_per_query: float,
    neg_frac: float,
    prompt_words: int,
    resolution: int,
    max_instances_per_query: int,
    seed: int,
):
    """A Sam3FalconBatch shaped like the real ones: `num_queries` queries spread
    over `num_images` images, a `neg_frac` share of them negatives, positives
    with ~Poisson(instances_per_query) (>= 1) rectangular instances."""
    rng = np.random.default_rng(seed)
    R = resolution
    per_image = [
        num_queries // num_images + (i < num_queries % num_images)
        for i in range(num_images)
    ]
    samples = []
    for i, nq in enumerate(per_image):
        image = rng.integers(0, 255, (3, R, R), dtype=np.uint8)
        texts, counts, boxes, bits = [], [], [], []
        for _ in range(nq):
            words = rng.choice(WORDS, size=max(1, int(rng.poisson(prompt_words))))
            texts.append(" ".join(words) + f" {rng.integers(1e9)}")  # unique per query
            if rng.random() < neg_frac:
                counts.append(0)
                continue
            n = int(
                min(max(1, rng.poisson(instances_per_query)), max_instances_per_query)
            )
            counts.append(n)
            for _ in range(n):
                w, h = rng.integers(8, R // 3, size=2)
                x0, y0 = rng.integers(0, R - w), rng.integers(0, R - h)
                m = np.zeros((R, R), np.uint8)
                m[y0 : y0 + h, x0 : x0 + w] = 1
                bits.append(pack_masks(m))
                boxes.append([(x0 + w / 2) / R, (y0 + h / 2) / R, w / R, h / R])
        samples.append(
            Sam3FalconSample(
                __key__=f"synthetic/{seed}_{i}",
                __restore_key__=(),
                image=image,
                texts=texts,
                num_instances=np.asarray(counts, np.int64),
                boxes=np.asarray(boxes, np.float32).reshape(-1, 4),
                mask_bits=(
                    np.stack(bits) if bits else np.zeros((0, R, (R + 7) // 8), np.uint8)
                ),
                source="synthetic",
                original_size=(R, R),
            )
        )
    return collate_samples(samples)


# ---------------------------------------------------------------------------
# Model / loss from the training config
# ---------------------------------------------------------------------------


def load_cfg(overrides: List[str]):
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    with contextlib.suppress(ValueError):
        register_omegaconf_resolvers()
    GlobalHydra.instance().clear()
    initialize_config_module("sam3.train", version_base="1.2")
    return compose(
        config_name=CONFIG,
        overrides=[
            "paths.experiment_log_dir=/tmp/sam3_bench",
            "trainer.model.pe_checkpoint=null",
        ]
        + overrides,
    )


def build(cfg, *, compile_vision: bool, device):
    import fnmatch

    from hydra.utils import instantiate

    model_cfg = cfg.trainer.model.copy()
    model_cfg.compile_vision = compile_vision
    model = instantiate(model_cfg, _convert_="all")
    for name, p in model.named_parameters():
        if any(fnmatch.fnmatch(name, pat) for pat in cfg.trainer.frozen_param_patterns):
            p.requires_grad_(False)
    model.batch_matching_over_layers = bool(cfg.trainer.batch_matching_over_layers)
    model.to(device).train()
    loss = instantiate(cfg.trainer.loss, _convert_="all").to(device)
    clipper = instantiate(cfg.trainer.optim.gradient_clip)
    return model, loss, clipper


def set_act_ckpt(model, off: List[str]):
    from sam3.train.model_init import set_activation_checkpointing

    set_activation_checkpointing(model, off)


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------


def train_step(model, loss_fn, clipper, opt, raw, device, amp_dtype):
    from sam3.train.streaming_trainer import CORE_LOSS_KEY

    m = model.module if hasattr(model, "module") else model
    batch = raw.to_batched_datapoint(device)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model(batch)
        losses = loss_fn(out, [m.back_convert(x) for x in batch.find_targets])
    core = losses[CORE_LOSS_KEY]
    core.backward()
    clipper(model=model)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return core.detach()


def run_config(
    args, model, loss_fn, clipper, opt, raw, device, profile_dir=None
) -> Dict:
    amp = torch.bfloat16 if args.amp == "bf16" else None
    raw = raw.pin_memory()
    try:
        for _ in range(args.warmup):
            train_step(model, loss_fn, clipper, opt, raw, device, amp)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times, losses = [], []
        for _ in range(args.steps):
            t0 = time.perf_counter()
            losses.append(train_step(model, loss_fn, clipper, opt, raw, device, amp))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        res = {
            "status": "ok",
            "step_s_median": float(np.median(times)),
            "step_s_min": float(np.min(times)),
            "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "loss_last": float(losses[-1]),
        }
        res["images_per_s"] = raw.images.shape[0] / res["step_s_median"]
        res["queries_per_s"] = raw.num_queries / res["step_s_median"]
        if profile_dir:
            res["profile"] = profile(
                model, loss_fn, clipper, opt, raw, device, amp, profile_dir
            )
        return res
    except torch.OutOfMemoryError as e:
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {"status": "oom", "error": str(e).splitlines()[0][:200]}


def profile(model, loss_fn, clipper, opt, raw, device, amp, out_dir) -> str:
    from torch.profiler import profile as tprofile, ProfilerActivity

    os.makedirs(out_dir, exist_ok=True)
    with tprofile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        for _ in range(2):
            train_step(model, loss_fn, clipper, opt, raw, device, amp)
        torch.cuda.synchronize()
    tag = f"rank{dist.get_rank() if dist.is_initialized() else 0}_{int(time.time())}"
    trace = os.path.join(out_dir, f"trace_{tag}.json")
    prof.export_chrome_trace(trace)
    with open(os.path.join(out_dir, f"top_ops_{tag}.txt"), "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
    return trace


# ---------------------------------------------------------------------------
# Numerics mode
# ---------------------------------------------------------------------------


def numerics(args, device):
    """Loss terms for one batch under each option, against a reference
    (fp32, per-layer matching, dense masks, bilinear target sampling)."""
    from hydra.utils import instantiate

    cfg = load_cfg(args.override)
    model, _, _ = build(cfg, compile_vision=False, device=device)
    losses = {}
    for ts in ("bilinear", "nearest"):
        lc = cfg.trainer.loss.copy()
        for fn in lc.loss_fns_find:
            if fn._target_.endswith("Masks"):
                fn.target_sampling = ts
        losses[ts] = instantiate(lc, _convert_="all").to(device)
    raw = synth_batch(
        num_images=args.images[0],
        num_queries=args.queries[0],
        instances_per_query=args.instances_per_query,
        neg_frac=args.neg_frac,
        prompt_words=args.prompt_words,
        resolution=1008,
        max_instances_per_query=200,
        seed=0,
    )
    batch = raw.to_batched_datapoint(device)
    variants = {
        "reference (fp32)": dict(bm=False, mm=False, ts="bilinear", amp=None),
        "batch_matching_over_layers": dict(bm=True, mm=False, ts="bilinear", amp=None),
        "masks_for_matched_queries_only": dict(
            bm=False, mm=True, ts="bilinear", amp=None
        ),
        "target_sampling=nearest": dict(bm=False, mm=False, ts="nearest", amp=None),
        "bf16 autocast": dict(bm=False, mm=False, ts="bilinear", amp=torch.bfloat16),
        "config (bf16, all flags)": dict(
            bm=True, mm=True, ts="nearest", amp=torch.bfloat16
        ),
    }
    results = {}
    for name, v in variants.items():
        model.batch_matching_over_layers = v["bm"]
        model.masks_for_matched_queries_only = v["mm"]
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        # Grad stays on: under no_grad the ViT MLP takes the bf16 inference kernel.
        with torch.autocast(
            "cuda", dtype=v["amp"] or torch.float32, enabled=v["amp"] is not None
        ):
            out = model(batch)
            ls = losses[v["ts"]](
                out, [model.back_convert(x) for x in batch.find_targets]
            )
        results[name] = {
            k: float(t)
            for k, t in ls.items()
            if isinstance(t, torch.Tensor) and t.numel() == 1
        }
        del out, ls
        torch.cuda.empty_cache()
    ref = results["reference (fp32)"]
    print(f"\n{'variant':34s} {'max rel diff':>12s}  worst term          core_loss")
    for name, r in results.items():
        diffs = {k: abs(r[k] - ref[k]) / max(abs(ref[k]), 1e-8) for k in ref}
        worst = max(diffs, key=diffs.get)
        print(f"{name:34s} {diffs[worst]:12.3e}  {worst:18s} {r['core_loss']:.6f}")
    print(
        "\nExpected: batch_matching_over_layers ~0 (same assignments); "
        "masks_for_matched_queries_only ~1e-6; target_sampling=nearest small (different sampling); "
        "bf16 ~1e-2."
    )
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps({"mode": "numerics", "results": results}) + "\n")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ints = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    ap.add_argument("--mode", choices=["sweep", "numerics"], default="sweep")
    ap.add_argument(
        "--queries", type=ints, default=[128, 256], help="queries per GPU step"
    )
    ap.add_argument("--images", type=ints, default=[8, 16], help="images per GPU step")
    ap.add_argument(
        "--instances-per-query",
        type=float,
        default=5.0,
        help="mean instances per positive query",
    )
    ap.add_argument(
        "--neg-frac", type=float, default=0.3, help="share of negative queries"
    )
    ap.add_argument("--prompt-words", type=int, default=4, help="mean words per prompt")
    ap.add_argument(
        "--matched-masks",
        type=ints,
        default=[0, 1],
        help="masks_for_matched_queries_only values",
    )
    ap.add_argument(
        "--compile", type=ints, default=[0], help="compile_vision values (0,1)"
    )
    ap.add_argument(
        "--ckpt-off",
        nargs="*",
        default=["none"],
        help="act-ckpt parts to turn off, one setting per arg, parts joined by '+': "
        "none | vit | text | seg_head | geometry, e.g. --ckpt-off none seg_head seg_head+text",
    )
    ap.add_argument("--amp", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument(
        "--find-unused",
        type=int,
        default=0,
        help="DDP find_unused_parameters (torchrun only)",
    )
    ap.add_argument(
        "--profile-dir",
        default=None,
        help="write a chrome trace + top-ops table per config",
    )
    ap.add_argument("--out", default=None, help="append results as JSONL")
    ap.add_argument(
        "--override", nargs="*", default=[], help="extra Hydra overrides for the config"
    )
    args = ap.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if distributed:
        dist.init_process_group("nccl")
    elif not dist.is_initialized():  # Sam3LossWrapper all-reduces the box count
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29400 + os.getpid() % 1000))
        dist.init_process_group("nccl", rank=0, world_size=1)
    rank = dist.get_rank()

    if args.mode == "numerics":
        if rank == 0:
            numerics(args, device)
        return

    cfg = load_cfg(args.override)
    rows = []
    for compile_vision in args.compile:
        model, loss_fn, clipper = build(
            cfg, compile_vision=bool(compile_vision), device=device
        )
        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                find_unused_parameters=bool(args.find_unused),
                gradient_as_bucket_view=True,
            )
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-5, fused=True
        )
        m = model.module if distributed else model
        for q, b, mm, off in itertools.product(
            args.queries, args.images, args.matched_masks, args.ckpt_off
        ):
            if b > q:
                continue
            off_parts = [] if off == "none" else off.split("+")
            set_act_ckpt(model, off_parts)
            m.masks_for_matched_queries_only = bool(mm)
            raw = synth_batch(
                num_images=b,
                num_queries=q,
                instances_per_query=args.instances_per_query,
                neg_frac=args.neg_frac,
                prompt_words=args.prompt_words,
                resolution=1008,
                max_instances_per_query=200,
                seed=1000 * rank + q + b,
            )
            conf = {
                "queries": q,
                "images": b,
                "instances": int(raw.boxes.shape[0]),
                "matched_masks": mm,
                "compile_vision": compile_vision,
                "ckpt_off": off,
                "amp": args.amp,
                "world_size": dist.get_world_size(),
                "gpu": torch.cuda.get_device_name(device),
            }
            res = run_config(
                args, model, loss_fn, clipper, opt, raw, device, args.profile_dir
            )
            row = {**conf, **res}
            rows.append(row)
            if rank == 0:
                print(_fmt(row), flush=True)
                if args.out:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
        del model, opt
        torch.cuda.empty_cache()
    if rank == 0:
        print("\n" + _HEADER)
        for r in rows:
            print(_fmt(r))
    dist.destroy_process_group()


_HEADER = (
    f"{'Q':>5} {'img':>4} {'inst':>5} {'mm':>2} {'cmp':>3} {'ckpt_off':>14} "
    f"{'step_s':>7} {'q/s':>7} {'img/s':>6} {'alloc':>6} {'reserv':>6} status"
)


def _fmt(r):
    if r.get("status") != "ok":
        return (
            f"{r['queries']:>5} {r['images']:>4} {r['instances']:>5} {r['matched_masks']:>2} "
            f"{r['compile_vision']:>3} {r['ckpt_off']:>14} {'':>7} {'':>7} {'':>6} {'':>6} {'':>6} {r.get('status')}"
        )
    return (
        f"{r['queries']:>5} {r['images']:>4} {r['instances']:>5} {r['matched_masks']:>2} "
        f"{r['compile_vision']:>3} {r['ckpt_off']:>14} {r['step_s_median']:7.3f} {r['queries_per_s']:7.1f} "
        f"{r['images_per_s']:6.2f} {r['peak_alloc_gib']:6.1f} {r['peak_reserved_gib']:6.1f} ok"
    )


if __name__ == "__main__":
    main()
