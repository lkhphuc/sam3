# pyre-unsafe
"""Falcon Perception v1.5 annotation-schema helpers, for training SAM3 on that data.

The perception tars store, per image, a JSON list of task objects
(`docs/V15_ANNOTATION_SCHEMA.md` in torchtitan-internal). SAM3 needs only the
referring-segmentation part of it: expression -> list of COCO RLE masks, the
image's negative expressions, expression alternatives (paraphrases,
translations) and expression attributes (level, lang).

The functions below are copied from torchtitan-internal at commit 1ca4fa3
(`torchtitan/energon_datasets/dataset_utils.py` and `task_processors.py`) rather
than imported: importing that package pulls in `torchtitan/__init__`
(quantization) and loads a spaCy model at import time in every loader worker.
They are pure functions over the JSON and must stay behaviourally identical to
the originals; `test/test_falcon_schema.py` checks parity when torchtitan is
importable. Keep the two in sync when the schema changes.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Task-object names (the "task" field of each JSON object).
REFERRING_SEGMENTATION = "referring_segmentation"
NEGATIVE_EXPRESSIONS = "negative_expressions"
EXPRESSION_ALTERNATIVES = "expression_alternatives"
EXPRESSION_ATTRIBUTES = "expression_attributes"


# ---------------------------------------------------------------------------
# Level / lang filters and alternatives (dataset_utils.py:347-418)
# ---------------------------------------------------------------------------


def parse_attribute_filter(value) -> Optional[List[str]]:
    """A `level` / `lang` subflavor as a list of strings; None when unset."""
    if value is None:
        return None
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(v).strip().lower() for v in values]


def _attribute_passes(value, allowed: Optional[List[str]]) -> bool:
    if allowed is None:
        return True
    values = value if isinstance(value, (list, tuple)) else [value]
    values = [str(v).lower() for v in values if v not in (None, "")]
    return any(v in allowed for v in values) if values else "none" in allowed


@dataclass
class AttributeFilter:
    """The `level` / `lang` selection of one source (None = no filter)."""

    level: Optional[List[str]] = None
    lang: Optional[List[str]] = None


def _string_passes(attrs: dict, config: AttributeFilter) -> bool:
    return _attribute_passes(attrs.get("level"), config.level) and _attribute_passes(
        attrs.get("lang"), config.lang
    )


def apply_expression_attributes(
    task_objects: List[dict],
    negative_samples: List[tuple],
    alternatives: dict,
    attributes: dict,
    config: AttributeFilter,
) -> Tuple[List[dict], List[tuple]]:
    """Filter expressions by level / language and pick each query's text.

    An expression's candidates are its key and its alternatives; an alternative
    inherits the attributes it does not set from its key. The expression is
    kept if a candidate passes, and its text is drawn uniformly from the
    passing candidates. An alternative listed under two keys, or equal to
    another key, could name either expression's objects, so it is never asked.
    """
    seen: Dict[str, int] = {}
    for alts in alternatives.values():
        for alt in set(alts):
            seen[alt] = seen.get(alt, 0) + 1
    keys = {k for meta in task_objects for k in meta if k != "task"}
    ambiguous = {alt for alt, n in seen.items() if n > 1} | (set(seen) & keys)
    alternatives = {
        key: [alt for alt in alts if alt not in ambiguous or alt == key]
        for key, alts in alternatives.items()
    }
    filtered = []
    for meta in task_objects:
        out = {}
        for key, value in meta.items():
            if key == "task":
                out[key] = value
                continue
            own = attributes.get(key, {})
            candidates = [(key, own)] + [
                (alt, {**own, **attributes.get(alt, {})})
                for alt in alternatives.get(key, [])
            ]
            passing = [
                text for text, attrs in candidates if _string_passes(attrs, config)
            ]
            if not passing:
                continue
            text = passing[0] if len(passing) == 1 else random.choice(passing)
            out[text if text not in out else key] = value
        filtered.append(out)
    negatives = [
        (k, v)
        for k, v in negative_samples
        if _string_passes(attributes.get(k, {}), config)
    ]
    return filtered, negatives


def clean_expression(expression: str) -> str:
    """Strip one leading article / determiner (task_processors.py:226)."""
    check_start = [
        "its ",
        "few ",
        "his ",
        "her ",
        "their ",
        "some ",
        "the ",
        "a ",
        "an ",
    ]
    for start in check_start:
        if expression.startswith(start):
            expression = expression[len(start) :]
    return expression


# ---------------------------------------------------------------------------
# RLE straight onto a target grid (task_processors.py:1513-1612)
# ---------------------------------------------------------------------------


def _nearest_indices(
    source_h: int, source_w: int, target_h: int, target_w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Source rows / columns sampled for each target row / column (ascending)."""
    return (
        np.round(np.linspace(0, source_h - 1, target_h)).astype(np.intp),
        np.round(np.linspace(0, source_w - 1, target_w)).astype(np.intp),
    )


def _rle_counts(counts) -> np.ndarray:
    """Run lengths of a COCO RLE: the compressed string decoded (pycocotools'
    rleFrString, vectorised), or an uncompressed list as is."""
    if not isinstance(counts, (str, bytes)):
        return np.asarray(counts, dtype=np.int64)
    a = (
        np.frombuffer(
            counts.encode("ascii") if isinstance(counts, str) else counts, np.uint8
        ).astype(np.int64)
        - 48
    )
    if not a.size:
        return a
    ends = np.flatnonzero((a & 0x20) == 0)
    starts = np.r_[0, ends[:-1] + 1]
    k = np.arange(a.size) - np.repeat(starts, ends - starts + 1)
    x = np.add.reduceat((a & 0x1F) << (5 * k), starts)
    neg = (a[ends] & 0x10) != 0
    x[neg] -= np.int64(1) << (5 * (k[ends[neg]] + 1))
    x[2::2] = np.cumsum(x[2::2])
    x[1::2] = np.cumsum(x[1::2])
    return x


def _rle_to_grid(rle: dict, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """The RLE mask sampled at (rows[i], cols[j]) as a [len(rows), len(cols)] uint8,
    equal to cocomask.decode(rle)[rows][:, cols], without the full-resolution decode."""
    source_h = int(rle["size"][0])
    counts = _rle_counts(rle["counts"])
    out = np.zeros((len(rows), len(cols)), np.uint8)
    end = np.cumsum(counts)
    run_start, run_end = (end - counts)[1::2], end[1::2]
    keep = run_end > run_start
    run_start, run_end = run_start[keep], run_end[keep]
    if not run_start.size or not source_h:
        return out
    first_col, last_col = run_start // source_h, (run_end - 1) // source_h
    n_cols = last_col - first_col + 1
    run = np.repeat(np.arange(run_start.size), n_cols)
    col = (
        np.repeat(first_col, n_cols)
        + np.arange(n_cols.sum())
        - np.repeat(np.cumsum(n_cols) - n_cols, n_cols)
    )
    row0 = np.maximum(run_start[run] - col * source_h, 0)
    row1 = np.minimum(run_end[run] - col * source_h, source_h)
    j0, j1 = np.searchsorted(cols, col, "left"), np.searchsorted(cols, col, "right")
    i0, i1 = np.searchsorted(rows, row0, "left"), np.searchsorted(rows, row1, "left")
    hit = (j1 > j0) & (i1 > i0)
    j0, j1, i0, i1 = j0[hit], j1[hit], i0[hit], i1[hit]
    if not j0.size:
        return out
    reps = j1 - j0  # > 1 only when upsampling repeats a source column
    j = (
        np.repeat(j0, reps)
        + np.arange(reps.sum())
        - np.repeat(np.cumsum(reps) - reps, reps)
    )
    i0, i1 = np.repeat(i0, reps), np.repeat(i1, reps)
    jlo, jhi, ilo, ihi = j.min(), j.max() + 1, i0.min(), i1.max()
    diff = np.zeros((ihi - ilo + 1, jhi - jlo), np.int32)
    np.add.at(diff, (i0 - ilo, j - jlo), 1)
    np.add.at(diff, (i1 - ilo, j - jlo), -1)
    out[ilo:ihi, jlo:jhi] = np.cumsum(diff, axis=0)[:-1] > 0
    return out


def decode_masks_resized(rles: List[dict], width: int, height: int) -> List[np.ndarray]:
    """Each RLE rasterised (nearest neighbour) onto a height x width uint8 grid."""
    index_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    masks = []
    for rle in rles:
        source_h, source_w = (int(v) for v in rle["size"])
        idx = index_cache.get((source_h, source_w))
        if idx is None:
            idx = _nearest_indices(source_h, source_w, height, width)
            index_cache[(source_h, source_w)] = idx
        masks.append(_rle_to_grid(rle, *idx))
    return masks


# ---------------------------------------------------------------------------
# Sample parsing for SAM3
# ---------------------------------------------------------------------------


@dataclass
class SegQuery:
    """One SAM3 find query: a text prompt and all of its instance masks (RLE)."""

    text: str
    rles: List[dict] = field(default_factory=list)  # [] for a negative

    @property
    def is_negative(self) -> bool:
        return not self.rles


def _is_rle(value: Any) -> bool:
    return isinstance(value, dict) and "counts" in value and "size" in value


def parse_segmentation_queries(
    metadata: Any,
    attribute_filter: AttributeFilter,
    *,
    strip_articles: bool = True,
) -> Tuple[List[SegQuery], Dict[str, int]]:
    """The referring-segmentation queries and negatives of one perception JSON.

    Reads only `referring_segmentation` objects (expression -> list of RLE),
    `negative_expressions`, `expression_alternatives` and
    `expression_attributes`. Detection, pointing, exemplar and caption objects
    are ignored: a grounding file often holds a box-only DET object next to the
    ref-seg object for the same phrase, which must never be used as a mask
    target.

    An empty list on a ref-seg key is dropped (never a negative), as in the
    Falcon loader: only `negative_expressions` marks an absent expression.
    A ref-seg value that is not a list of RLE dicts is dropped and counted.

    Returns the queries (positives first, in file order, then negatives) and
    per-reason drop counts.
    """
    stats = {"empty_positive": 0, "non_rle_value": 0, "positive_also_negative": 0}
    metadata_list = metadata if isinstance(metadata, list) else [metadata]
    seg_objects, negatives_obj = [], None
    alternatives: dict = {}
    attributes: dict = {}
    for meta in metadata_list:
        if not isinstance(meta, dict):
            continue
        name = str(meta.get("task", "")).lower().strip()
        if name == REFERRING_SEGMENTATION:
            seg_objects.append(meta)
        elif name == NEGATIVE_EXPRESSIONS:
            negatives_obj = meta
        elif name == EXPRESSION_ALTERNATIVES:
            alternatives.update({k: v for k, v in meta.items() if k != "task"})
        elif name == EXPRESSION_ATTRIBUTES:
            attributes.update(
                {k: v for k, v in meta.items() if k != "task" and isinstance(v, dict)}
            )
    negative_samples = (
        [(k, v) for k, v in negatives_obj.items() if k != "task"]
        if negatives_obj is not None
        else []
    )
    if (
        alternatives
        or attribute_filter.level is not None
        or attribute_filter.lang is not None
    ):
        seg_objects, negative_samples = apply_expression_attributes(
            seg_objects, negative_samples, alternatives, attributes, attribute_filter
        )

    queries: List[SegQuery] = []
    positive_texts = set()
    for meta in seg_objects:
        for key, value in meta.items():
            if key == "task":
                continue
            if not isinstance(value, list) or not value:
                stats["empty_positive"] += 1
                continue
            if not all(_is_rle(v) for v in value):
                stats["non_rle_value"] += 1
                continue
            text = clean_expression(key) if strip_articles else key
            queries.append(SegQuery(text=text, rles=list(value)))
            positive_texts.add(text)
    for key, _ in negative_samples:
        text = clean_expression(key) if strip_articles else key
        if text in positive_texts:
            # After article stripping a negative can collide with a positive
            # ("a dog" vs "the dog"); the positive wins.
            stats["positive_also_negative"] += 1
            continue
        queries.append(SegQuery(text=text, rles=[]))
    return queries, stats


# ---------------------------------------------------------------------------
# Bit-packed masks (falcon_taskencoder.py:52)
# ---------------------------------------------------------------------------


def pack_masks(masks: np.ndarray) -> np.ndarray:
    """[N, H, W] uint8 {0,1} -> [N, H, ceil(W/8)] uint8, MSB first along W."""
    if masks.dtype != np.uint8:
        raise ValueError(f"pack_masks expects uint8, got {masks.dtype}")
    return np.packbits(masks, axis=-1)


def unpack_masks(bits: np.ndarray, width: int) -> np.ndarray:
    return np.unpackbits(bits, axis=-1, count=width)
