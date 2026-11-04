"""Run the real SAM3 energon loader on the blend: throughput, batch stats,
host memory and rendered batches (CPU; needs the Energon fork and the mounts).

    python scripts/falcon_v15/inspect_loader.py --batches 200 --render 4 --out /tmp/loader_check
    # any training-config override, e.g. fewer workers or another blend:
    python scripts/falcon_v15/inspect_loader.py --batches 50 \
        trainer.data.train.num_workers=4 paths.dataset_path=/path/to/blend.yaml

Prints per-batch queries / images / instances / negatives and the time to get
each batch, then a summary (batches/s, p95 wait, per-source image counts, host
RSS of this process + workers). Writes the first --render batches as PNGs:
each image with every positive query's masks in one colour and its text, the
negatives listed below it. Compare them with the source JSON by eye: masks on
the right objects, the text matching, nothing shifted by the resize / pad.
"""

import argparse
import collections
import contextlib
import os
import time

import numpy as np


def _rss_gib() -> float:
    """RSS of this process and its children (the loader workers), from /proc."""
    total = 0
    pids = [os.getpid()]
    try:
        children = (
            open(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read().split()
        )
        pids += [int(c) for c in children]
    except OSError:
        pass
    for pid in pids:
        try:
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except OSError:
            pass
    return total / 2**30


def render(batch, path, max_images=4):
    from PIL import Image, ImageDraw

    from sam3.train.data.falcon_schema import unpack_masks

    R = batch.resolution
    img_ids = batch.img_ids.numpy()
    counts = batch.num_boxes.numpy()
    starts = np.cumsum(counts) - counts
    masks = unpack_masks(batch.mask_bits.numpy(), R).astype(bool)
    boxes = batch.boxes.numpy()
    rng = np.random.default_rng(0)
    tiles = []
    for i in range(min(max_images, batch.images.shape[0])):
        rgb = batch.images[i].permute(1, 2, 0).numpy().astype(np.float32)
        labels, negs = [], []
        for qi in np.nonzero(img_ids == i)[0]:
            text = batch.texts[int(batch.text_ids[qi])]
            if counts[qi] == 0:
                negs.append(text)
                continue
            color = rng.integers(40, 255, 3)
            for j in range(starts[qi], starts[qi] + counts[qi]):
                rgb[masks[j]] = 0.45 * rgb[masks[j]] + 0.55 * color
                cx, cy, w, h = boxes[j] * R
                labels.append(
                    (
                        cx - w / 2,
                        cy - h / 2,
                        cx + w / 2,
                        cy + h / 2,
                        text,
                        tuple(int(c) for c in color),
                    )
                )
        tile = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(tile)
        for x0, y0, x1, y1, text, color in labels:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            draw.text((x0 + 2, y0 + 2), text[:60], fill=color)
        canvas = Image.new("RGB", (R, R + 16 * (len(negs) + 1)), "black")
        canvas.paste(tile, (0, 0))
        d2 = ImageDraw.Draw(canvas)
        d2.text((4, R + 2), f"{batch.sources[i]} | {batch.__key__[i]}", fill="white")
        for k, t in enumerate(negs):
            d2.text((4, R + 16 * (k + 1)), f"negative: {t[:100]}", fill=(255, 120, 120))
        tiles.append(canvas)
    w = sum(t.width for t in tiles)
    h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (w, h), "black")
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--render", type=int, default=4, help="render the first N batches")
    ap.add_argument("--out", default="/tmp/sam3_loader_check")
    ap.add_argument(
        "overrides", nargs="*", help="Hydra overrides of the training config"
    )
    args = ap.parse_args()

    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate

    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    with contextlib.suppress(ValueError):
        register_omegaconf_resolvers()
    initialize_config_module("sam3.train", version_base="1.2")
    cfg = compose(
        config_name="configs/falcon_v15/sam3_falcon_v15_seg.yaml",
        overrides=[f"paths.experiment_log_dir={args.out}"] + args.overrides,
    )
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    loader = instantiate(cfg.trainer.data.train, _convert_="all")
    it = iter(loader)
    waits, rows = [], []
    sources = collections.Counter()
    first = None
    for b in range(args.batches):
        t = time.perf_counter()
        batch = next(it)
        wait = time.perf_counter() - t
        if first is None:
            first = time.perf_counter() - t0
        waits.append(wait)
        n_neg = int((batch.num_boxes == 0).sum())
        rows.append(
            (
                batch.num_queries,
                batch.images.shape[0],
                int(batch.boxes.shape[0]),
                n_neg,
                len(batch.texts),
            )
        )
        sources.update(batch.sources)
        if b < 20 or b % 20 == 0:
            print(
                f"batch {b:4d}: wait {wait:6.3f}s queries {batch.num_queries:4d} images {batch.images.shape[0]:3d} "
                f"instances {batch.boxes.shape[0]:5d} negatives {n_neg:4d} unique texts {len(batch.texts):4d} "
                f"rss {_rss_gib():6.1f} GiB",
                flush=True,
            )
        if b < args.render:
            render(batch, os.path.join(args.out, f"batch_{b:03d}.png"))
    a = np.asarray(rows)
    w = np.asarray(waits[1:]) if len(waits) > 1 else np.asarray(waits)
    print(
        f"\nfirst batch after {first:.1f}s (buffer fill); then {1 / w.mean():.2f} batches/s, wait p50 {np.median(w):.3f}s p95 {np.percentile(w, 95):.3f}s max {w.max():.3f}s"
    )
    print(
        f"per batch mean: queries {a[:, 0].mean():.1f} images {a[:, 1].mean():.1f} instances {a[:, 2].mean():.1f} negatives {a[:, 3].mean():.1f}"
    )
    print(
        f"budget use: queries {a[:, 0].max()}/{cfg.scratch.max_queries_per_batch} max, instances {a[:, 2].max()}/{cfg.scratch.max_instances_per_batch} max, images {a[:, 1].max()}/{cfg.scratch.max_images_per_batch} max"
    )
    print(f"host RSS (this process + workers): {_rss_gib():.1f} GiB")
    print("images per source:")
    for s, n in sources.most_common():
        print(f"  {n:6d}  {s}")
    print(
        f"\nrendered batches in {args.out}; encoder stats are logged by each worker every 2000 samples"
    )


if __name__ == "__main__":
    main()
