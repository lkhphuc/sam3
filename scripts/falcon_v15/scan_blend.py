"""Scan the SAM3 segmentation blend's annotations straight from the tars (CPU only).

For every source of the blend it reads up to --max-samples perception JSONs
(spread over up to --max-tars tars) and reports:

  masks     share of samples with >= 1 referring_segmentation query whose values
            are RLE masks, and how many ref-seg values are not RLE (box-only
            data would show up here and be dropped by the SAM3 loader);
  queries   positives / negatives per image (p50 / p95 / max), so the per-GPU
            query budget can be sized;
  dense     share of positive queries (and of masks) with more than
            --max-instances instances, which the loader drops;
  tokens    prompt length in SAM3 BPE tokens (with start/end tokens), by level
            and language: p50 / p99 / max and the share above 32 and above the
            configured context, for every candidate text (keys, alternatives,
            negatives).

    python scripts/falcon_v15/scan_blend.py \
        sam3/train/configs/falcon_v15/datasets/sam3_falcon_v15_seg.yaml \
        --max-samples 2000 --workers 16 --out /tmp/scan_blend.json

Needs only the dataset mounts (no GPU, no energon). The metadata member of each
sample is found from the source's .nv-meta/dataset.yaml field_map (`metadata`
field, default `json`).
"""

import argparse
import collections
import glob
import json
import os
import random
import tarfile
from multiprocessing import Pool

import numpy as np
import yaml

_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        import pkg_resources

        from sam3.model.tokenizer_ve import SimpleTokenizer

        _TOKENIZER = SimpleTokenizer(
            bpe_path=pkg_resources.resource_filename(
                "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
            )
        )
    return _TOKENIZER


def _n_tokens(text: str) -> int:
    return len(_tokenizer().encode(text)) + 2  # + start / end of text


def _metadata_ext(path: str) -> str:
    try:
        with open(os.path.join(path, ".nv-meta", "dataset.yaml")) as f:
            fm = (yaml.safe_load(f) or {}).get("field_map", {})
        ext = fm.get("metadata", "json")
        return ext if isinstance(ext, str) else "json"
    except OSError:
        return "json"


def _tars(path: str):
    info = os.path.join(path, ".nv-meta", ".info.yaml")
    if os.path.exists(info):
        with open(info) as f:
            counts = (yaml.safe_load(f) or {}).get("shard_counts", {})
        if counts:
            return [os.path.join(path, t) for t in counts]
    return sorted(glob.glob(os.path.join(path, "**", "*.tar"), recursive=True))


def _is_rle(v) -> bool:
    return isinstance(v, dict) and "counts" in v and "size" in v


def scan_source(job):
    path, max_samples, max_tars, max_instances, seed = job
    rng = random.Random(seed)
    tars = _tars(path)
    rng.shuffle(tars)
    tars = tars[:max_tars]
    ext = "." + _metadata_ext(path)
    st = collections.Counter()
    pos_per_img, neg_per_img, inst_per_query = [], [], []
    tokens = collections.defaultdict(list)  # (level, lang) -> lengths
    per_tar = max(1, max_samples // max(1, len(tars)))
    for tar_path in tars:
        try:
            tf = tarfile.open(tar_path)
        except (OSError, tarfile.TarError) as e:
            st["tar_errors"] += 1
            st[f"tar_error:{type(e).__name__}"] += 1
            continue
        n = 0
        with tf:
            for member in tf:
                if n >= per_tar or st["samples"] >= max_samples:
                    break
                if not member.name.endswith(ext):
                    continue
                try:
                    meta = json.load(tf.extractfile(member))
                except (ValueError, OSError):
                    st["json_errors"] += 1
                    continue
                n += 1
                _scan_sample(
                    meta,
                    st,
                    pos_per_img,
                    neg_per_img,
                    inst_per_query,
                    tokens,
                    max_instances,
                )
        if st["samples"] >= max_samples:
            break
    return (
        path,
        dict(st),
        pos_per_img,
        neg_per_img,
        inst_per_query,
        {f"{k[0]}|{k[1]}": v for k, v in tokens.items()},
    )


def _scan_sample(
    meta, st, pos_per_img, neg_per_img, inst_per_query, tokens, max_instances
):
    st["samples"] += 1
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
        else:
            st[f"other_task:{task or '?'}"] += 1
    n_pos = 0
    for m in refseg:
        for key, value in m.items():
            if key == "task":
                continue
            if not isinstance(value, list) or not value:
                st["refseg_empty_value"] += 1
                continue
            if not all(_is_rle(v) for v in value):
                st["refseg_non_rle_value"] += 1
                continue
            st["positive_queries"] += 1
            st["masks"] += len(value)
            inst_per_query.append(len(value))
            if len(value) > max_instances:
                st["queries_over_max_instances"] += 1
                st["masks_in_queries_over_max_instances"] += len(value)
                continue
            n_pos += 1
            own = attributes.get(key, {})
            for text in [key] + list(alternatives.get(key, [])):
                attrs = {**own, **attributes.get(text, {})}
                tokens[
                    (str(attrs.get("level", "none")), str(attrs.get("lang", "none")))
                ].append(_n_tokens(text))
    for key in negatives:
        attrs = attributes.get(key, {})
        tokens[
            (f"neg:{attrs.get('level', 'none')}", str(attrs.get("lang", "none")))
        ].append(_n_tokens(key))
    st["samples_with_mask_query"] += int(n_pos > 0)
    st["negative_queries"] += len(negatives)
    pos_per_img.append(n_pos)
    neg_per_img.append(len(negatives))


def _pct(values, qs=(50, 95)):
    if not values:
        return "-"
    a = np.asarray(values)
    return "/".join(f"{np.percentile(a, q):.0f}" for q in qs) + f"/{a.max()}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("blend")
    ap.add_argument(
        "--max-samples", type=int, default=2000, help="JSONs read per source"
    )
    ap.add_argument("--max-tars", type=int, default=20, help="tars sampled per source")
    ap.add_argument(
        "--max-instances",
        type=int,
        default=200,
        help="loader's max_instances_per_query",
    )
    ap.add_argument(
        "--context", type=int, default=256, help="text context length of the run"
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="write the full results as JSON")
    args = ap.parse_args()

    with open(args.blend) as f:
        blend = yaml.safe_load(f)
    paths = [e["path"] for e in blend["splits"]["train"]["blend_epochized"]]
    jobs = [
        (p, args.max_samples, args.max_tars, args.max_instances, i)
        for i, p in enumerate(paths)
    ]
    with Pool(args.workers) as pool:
        results = pool.map(scan_source, jobs)

    all_tokens = collections.defaultdict(list)
    print(
        f"{'source':58s} {'samples':>7} {'w/mask%':>7} {'nonRLE':>6} {'pos/img p50/95/max':>18} "
        f"{'neg/img':>11} {'>200 q%':>7} {'>200 m%':>7} {'tok p50/95/max':>14}"
    )
    out = {}
    for path, st, pos, neg, inst, toks in results:
        s = max(st.get("samples", 0), 1)
        flat = [t for v in toks.values() for t in v]
        for k, v in toks.items():
            all_tokens[k].extend(v)
        name = path.replace("/mnt/falcon-perception/", "")[-58:]
        q = max(st.get("positive_queries", 0), 1)
        m = max(st.get("masks", 0), 1)
        print(
            f"{name:58s} {st.get('samples', 0):7d} {100 * st.get('samples_with_mask_query', 0) / s:7.1f} "
            f"{st.get('refseg_non_rle_value', 0):6d} {_pct(pos):>18} {_pct(neg):>11} "
            f"{100 * st.get('queries_over_max_instances', 0) / q:7.2f} "
            f"{100 * st.get('masks_in_queries_over_max_instances', 0) / m:7.2f} {_pct(flat):>14}"
        )
        out[path] = {
            "stats": st,
            "pos_per_img": _pct(pos),
            "neg_per_img": _pct(neg),
            "inst_per_query": _pct(inst, (50, 95, 99)),
            "tokens": _pct(flat, (50, 99)),
        }
        if not st.get("samples"):
            print(f"   !! no sample read from {path}: {st}")
        elif st.get("samples_with_mask_query", 0) == 0:
            print(f"   !! no mask query in {path}: {st}")

    print(
        f"\nPrompt length in tokens (incl. start/end) by level|lang, context {args.context}:"
    )
    print(
        f"{'level|lang':24s} {'n':>8} {'p50':>5} {'p99':>5} {'max':>5} {'>32 %':>7} {f'>{args.context} %':>8}"
    )
    for k in sorted(all_tokens, key=lambda k: -len(all_tokens[k])):
        a = np.asarray(all_tokens[k])
        print(
            f"{k:24s} {a.size:8d} {np.percentile(a, 50):5.0f} {np.percentile(a, 99):5.0f} {a.max():5d} "
            f"{100 * (a > 32).mean():7.2f} {100 * (a > args.context).mean():8.3f}"
        )
    if args.out:
        out["_tokens_by_level_lang"] = {
            k: _pct(v, (50, 99)) for k, v in all_tokens.items()
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
