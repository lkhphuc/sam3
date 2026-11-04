"""Synthetic Falcon Perception v1.5 samples for the SAM3 loader tests."""

import io
from types import SimpleNamespace

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image


def rle(mask: np.ndarray) -> dict:
    enc = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    enc["counts"] = enc["counts"].decode("ascii")
    return enc


def rect_mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def image_bytes(h, w, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def perception_metadata(h=300, w=400):
    """Two positives (one with 2 instances), one DET object that must be
    ignored, one empty ref-seg key, two negatives, alternatives and attributes."""
    dog1 = rect_mask(h, w, 10, 60, 20, 100)
    dog2 = rect_mask(h, w, 150, 250, 200, 380)
    cat = rect_mask(h, w, 100, 140, 50, 90)
    return [
        {
            "task": "object detection",
            "the dog": [{"x1": 5, "y1": 3, "x2": 25, "y2": 20}],
        },
        {
            "task": "referring_segmentation",
            "the dog": [rle(dog1), rle(dog2)],
            "a sleeping cat on the left": [rle(cat)],
            "the unicorn": [],
        },
        {"task": "negative_expressions", "a bicycle": [], "a boat": []},
        {
            "task": "expression_alternatives",
            "a sleeping cat on the left": ["القط النائم"],
        },
        {
            "task": "expression_attributes",
            "the dog": {"level": "0", "lang": "en"},
            "a sleeping cat on the left": {"level": "3", "lang": "en"},
            "القط النائم": {"lang": "ar"},
        },
    ], {"dog": [dog1, dog2], "cat": [cat]}


def raw_sample(key="src/000001", h=300, w=400, subflavors=None):
    metadata, masks = perception_metadata(h, w)
    sample = SimpleNamespace(
        __key__=key,
        __restore_key__=("Webdataset", key),
        __subflavors__=subflavors or {"task": "referring_segmentation"},
        images=[image_bytes(h, w)],
        metadata=metadata,
    )
    return sample, masks
