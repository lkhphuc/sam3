"""Build a held-out Falcon v1.5 segmentation eval set in SA-Co/Gold format.

Reads the `val` split tars (.nv-meta/split.yaml) of every source of the blend,
never the training tars, and writes

    <out>/images/<source>__<key>.<ext>   the original image bytes
    <out>/gt.json                        COCO-style GT, one "image" per
                                         (image, phrase) pair, as the SA-Co/Gold
                                         loaders and the cgF1 evaluator expect

Pairs follow the training data semantics (sam3.train.data.falcon_energon):
referring_segmentation keys (leading article stripped) with their RLE masks,
file negatives with no annotation, queries with more than --max-instances
masks left out. Each pair keeps `source`, `level`, `lang` and `num_instances`
for slicing the metrics (scripts/falcon_v15/cgf1_by_slice.py). Phrases are the
canonical keys; --alternatives adds each expression_alternatives text as its
own pair with the same masks.

    python scripts/falcon_v15/make_falcon_eval_set.py \
        sam3/train/configs/falcon_v15/datasets/sam3_falcon_v15_seg.yaml \
        --out /mnt/falcon-perception/sam3_eval/falcon_v15_heldout --max-images-per-source 300

A source without a val split is skipped (listed at the end): prepare one with
torchtitan-internal's energon_prepare_perception.py --val-tar, or pass
--allow-train-tars to use training tars (the metrics then measure fit, not
generalisation).
"""

import argparse
import io
import json
import os
import random
import sys
import tarfile

import numpy as np
import pycocotools.mask as mask_utils
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from scan_blend import _metadata_ext, split_tars  # noqa: E402

from sam3.train.data.falcon_schema import (
    clean_expression,
    decode_masks_resized,
)  # noqa: E402


def _image_ext(path: str) -> str:
    try:
        with open(os.path.join(path, ".nv-meta", "dataset.yaml")) as f:
            fm = (yaml.safe_load(f) or {}).get("field_map", {})
        ext = fm.get("images", fm.get("image", "npy"))
        return ext if isinstance(ext, str) else "npy"
    except OSError:
        return "npy"


def _image_bytes(raw: bytes, ext: str):
    """(encoded image bytes, file extension, PIL image) of the sample's first image."""
    if ext != "npy":
        img = Image.open(io.BytesIO(raw))
        return raw, (img.format or ext).lower().replace("jpeg", "jpg"), img
    arr = np.load(io.BytesIO(raw), allow_pickle=True)
    first = arr[0] if (arr.dtype == object or arr.ndim == 1) and len(arr) else arr
    if isinstance(first, (bytes, bytearray, np.bytes_)):
        img = Image.open(io.BytesIO(bytes(first)))
        return bytes(first), (img.format or "png").lower().replace("jpeg", "jpg"), img
    if isinstance(first, np.ndarray) and first.dtype == np.uint8 and first.ndim == 1:
        b = first.tobytes()
        img = Image.open(io.BytesIO(b))
        return b, (img.format or "png").lower().replace("jpeg", "jpg"), img
    img = Image.fromarray(np.asarray(first))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "png", img


def _is_rle(v) -> bool:
    return isinstance(v, dict) and "counts" in v and "size" in v


def _pairs(meta, alternatives_too: bool, max_instances: int):
    """(text, rles, level, lang) per query of one perception JSON."""
    metas = meta if isinstance(meta, list) else [meta]
    refseg, negatives, alternatives, attributes = [], {}, {}, {}
    for m in metas:
        if not isinstance(m, dict):
            continue
        task = str(m.get("task", "")).lower().strip()
        if task == "referring_segmentation":
            refseg.append(m)
        elif task == "negative_expressions":
            negatives.update({k: v for k, v in m.items() if k != "task"})
        elif task == "expression_alternatives":
            alternatives.update({k: v for k, v in m.items() if k != "task"})
        elif task == "expression_attributes":
            attributes.update(
                {k: v for k, v in m.items() if k != "task" and isinstance(v, dict)}
            )
    out, seen = [], set()
    for m in refseg:
        for key, value in m.items():
            if key == "task" or not isinstance(value, list) or not value:
                continue
            if not all(_is_rle(v) for v in value) or len(value) > max_instances:
                continue
            own = attributes.get(key, {})
            texts = [key] + (
                list(alternatives.get(key, [])) if alternatives_too else []
            )
            for text in texts:
                attrs = {**own, **attributes.get(text, {})}
                t = clean_expression(text)
                if t not in seen:
                    seen.add(t)
                    out.append(
                        (
                            t,
                            value,
                            str(attrs.get("level", "none")),
                            str(attrs.get("lang", "none")),
                        )
                    )
    for key in negatives:
        t = clean_expression(key)
        if t not in seen:
            seen.add(t)
            attrs = attributes.get(key, {})
            out.append(
                (
                    t,
                    [],
                    f"neg:{attrs.get('level', 'none')}",
                    str(attrs.get("lang", "none")),
                )
            )
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("blend")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images-per-source", type=int, default=300)
    ap.add_argument("--max-instances", type=int, default=200)
    ap.add_argument("--alternatives", action="store_true")
    ap.add_argument("--allow-train-tars", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.blend) as f:
        blend = yaml.safe_load(f)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    rng = random.Random(args.seed)
    images, annotations, skipped, per_source = [], [], [], {}
    for entry in blend["splits"]["train"]["blend_epochized"]:
        path = entry["path"]
        source = (
            path.rstrip("/").replace("/mnt/falcon-perception/", "").replace("/", "_")
        )
        tars = split_tars(path, "val") or []
        if not tars and args.allow_train_tars:
            tars = split_tars(path, "train") or []
        if not tars:
            skipped.append(path)
            continue
        rng.shuffle(tars)
        meta_ext, img_ext = "." + _metadata_ext(path), "." + _image_ext(path)
        n_img = 0
        for tar_path in tars:
            if n_img >= args.max_images_per_source:
                break
            with tarfile.open(tar_path) as tf:
                members = {}
                for mem in tf:
                    key, _, ext = mem.name.rpartition(".")
                    if "." + ext in (meta_ext, img_ext):
                        members.setdefault(key, {})["." + ext] = mem
                for key, mm in members.items():
                    if n_img >= args.max_images_per_source:
                        break
                    if meta_ext not in mm or img_ext not in mm:
                        continue
                    meta = json.load(tf.extractfile(mm[meta_ext]))
                    pairs = _pairs(meta, args.alternatives, args.max_instances)
                    if not any(rles for _, rles, _, _ in pairs):
                        continue
                    raw, ext, img = _image_bytes(
                        tf.extractfile(mm[img_ext]).read(), img_ext[1:]
                    )
                    w, h = img.size
                    fname = f"{source}__{key.replace('/', '_')}.{ext}"
                    with open(os.path.join(args.out, "images", fname), "wb") as f:
                        f.write(raw)
                    for text, rles, level, lang in pairs:
                        # Masks rasterised on the stored image grid; an empty
                        # one (outside the image) is not an instance.
                        masks = [m for m in decode_masks_resized(rles, w, h) if m.any()]
                        if rles and not masks:
                            continue
                        pair_id = len(images)
                        images.append(
                            {
                                "id": pair_id,
                                "file_name": fname,
                                "height": h,
                                "width": w,
                                "text_input": text,
                                "queried_category": 1,
                                "is_instance_exhaustive": True,
                                "is_pixel_exhaustive": True,
                                "source": source,
                                "level": level,
                                "lang": lang,
                                "num_instances": len(masks),
                            }
                        )
                        for m in masks:
                            enc = mask_utils.encode(np.asfortranarray(m))
                            x, y, bw, bh = mask_utils.toBbox(enc).tolist()
                            enc["counts"] = enc["counts"].decode("ascii")
                            annotations.append(
                                {
                                    "id": len(annotations) + 1,
                                    "image_id": pair_id,
                                    "category_id": 1,
                                    "bbox": [x, y, bw, bh],
                                    "area": float(mask_utils.area(enc)),
                                    "segmentation": enc,
                                    "iscrowd": 0,
                                }
                            )
                    n_img += 1
        per_source[source] = n_img
    gt = {
        "info": {
            "description": "Falcon v1.5 segmentation held-out pairs",
            "blend": args.blend,
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    with open(os.path.join(args.out, "gt.json"), "w") as f:
        json.dump(gt, f, ensure_ascii=False)
    n_neg = sum(1 for im in images if im["num_instances"] == 0)
    print(
        f"{len(images)} pairs ({n_neg} negative), {len(annotations)} masks, {sum(per_source.values())} images"
    )
    for s, n in sorted(per_source.items()):
        print(f"  {n:5d} images  {s}")
    if skipped:
        print(f"\nskipped (no val split; see --allow-train-tars): {len(skipped)}")
        for p in skipped:
            print(f"  {p}")


if __name__ == "__main__":
    main()
