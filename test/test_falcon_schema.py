import random

import numpy as np
import pycocotools.mask as mask_utils
import pytest

from falcon_fixtures import perception_metadata, rect_mask, rle
from sam3.train.data import falcon_schema as fs


def test_parse_keeps_only_refseg_and_file_negatives():
    meta, _ = perception_metadata()
    queries, stats = fs.parse_segmentation_queries(meta, fs.AttributeFilter())
    pos = {q.text: len(q.rles) for q in queries if not q.is_negative}
    neg = [q.text for q in queries if q.is_negative]
    # DET object ignored (box-only), articles stripped, empty ref-seg key dropped.
    assert len(pos) == 2 and pos["dog"] == 2
    assert {"sleeping cat on the left", "القط النائم"} & set(pos)
    assert sorted(neg) == ["bicycle", "boat"]
    assert stats["empty_positive"] == 1
    assert "unicorn" not in pos and "unicorn" not in neg


def test_level_and_lang_filters():
    meta, _ = perception_metadata()
    q, _ = fs.parse_segmentation_queries(meta, fs.AttributeFilter(level=["0"]))
    assert [x.text for x in q if not x.is_negative] == ["dog"]
    # Negatives have no attributes: they need "none" to pass a level filter.
    assert not [x for x in q if x.is_negative]
    random.seed(0)
    q, _ = fs.parse_segmentation_queries(meta, fs.AttributeFilter(lang=["ar"]))
    assert [x.text for x in q if not x.is_negative] == ["القط النائم"]


def test_alternative_drawn_from_candidates():
    meta, _ = perception_metadata()
    seen = set()
    for seed in range(30):
        random.seed(seed)
        q, _ = fs.parse_segmentation_queries(meta, fs.AttributeFilter())
        seen |= {x.text for x in q if len(x.rles) == 1}
    assert seen == {"sleeping cat on the left", "القط النائم"}


def test_negative_colliding_with_positive_is_dropped():
    m = rect_mask(20, 20, 2, 8, 2, 8)
    meta = [
        {"task": "referring_segmentation", "the dog": [rle(m)]},
        {"task": "negative_expressions", "a dog": []},
    ]
    q, stats = fs.parse_segmentation_queries(meta, fs.AttributeFilter())
    assert [(x.text, x.is_negative) for x in q] == [("dog", False)]
    assert stats["positive_also_negative"] == 1


@pytest.mark.parametrize(
    "src,dst", [((300, 400), (1008, 1008)), ((901, 37), (100, 333))]
)
def test_decode_resized_equals_full_decode_then_nearest(src, dst):
    rng = np.random.default_rng(0)
    h, w = src
    m = (rng.random((h, w)) > 0.7).astype(np.uint8)
    out = fs.decode_masks_resized([rle(m)], dst[1], dst[0])[0]
    rows, cols = fs._nearest_indices(h, w, dst[0], dst[1])
    ref = mask_utils.decode(rle(m))[rows][:, cols]
    np.testing.assert_array_equal(out, ref)


def test_pack_roundtrip():
    rng = np.random.default_rng(0)
    m = (rng.random((3, 17, 1010)) > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(fs.unpack_masks(fs.pack_masks(m), 1010), m)


def _torchtitan_functions(names):
    """Named top-level functions from a torchtitan-internal checkout, compiled
    from source (importing the package needs its full training stack). Looks at
    $TORCHTITAN_INTERNAL_DIR, then a sibling checkout of this repo."""
    import ast
    import os
    import types

    root = os.environ.get(
        "TORCHTITAN_INTERNAL_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "torchtitan-internal"),
    )
    base = os.path.join(root, "torchtitan", "energon_datasets")
    if not os.path.isdir(base):
        pytest.skip("no torchtitan-internal checkout")
    ns = {"np": np, "random": random, "Optional": None}
    for fname in ("task_processors.py", "dataset_utils.py"):
        tree = ast.parse(open(os.path.join(base, fname)).read())
        body = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        for n in body:
            n.returns = None
            for a in n.args.args + n.args.kwonlyargs:
                a.annotation = None
        exec(compile(ast.Module(body=body, type_ignores=[]), fname, "exec"), ns)
    missing = [n for n in names if n not in ns]
    assert not missing, f"not found in torchtitan-internal: {missing}"
    return types.SimpleNamespace(**{n: ns[n] for n in names})


def test_parity_with_torchtitan():
    """The copied helpers behave exactly like torchtitan-internal's."""
    tt = _torchtitan_functions(
        [
            "_nearest_indices",
            "_rle_counts",
            "_rle_to_grid",
            "_decode_masks_resized",
            "_attribute_passes",
            "_string_passes",
            "_apply_expression_attributes",
            "parse_attribute_filter",
            "clean_expression",
            "_batch_decode_masks",
        ]
    )
    rng = np.random.default_rng(1)
    for shape, target in [((120, 77), (50, 300)), ((640, 480), (1008, 1008))]:
        m = (rng.random(shape) > 0.6).astype(np.uint8)
        a = tt._decode_masks_resized([rle(m)], target)[0]
        b = fs.decode_masks_resized([rle(m)], *target)[0]
        np.testing.assert_array_equal(a, b)
    for text in ["the dog", "an apple", "their car", "dog", "a"]:
        assert tt.clean_expression(text) == fs.clean_expression(text)
    meta, _ = perception_metadata()
    alts = {"a sleeping cat on the left": ["القط النائم", "the dog"]}
    attrs = {"the dog": {"level": "0"}, "القط النائم": {"lang": "ar"}}
    for level, lang in [
        (None, None),
        (["0"], None),
        (["3", "none"], ["ar"]),
        (None, ["en"]),
    ]:
        cfg = fs.AttributeFilter(
            level=tt.parse_attribute_filter(level), lang=tt.parse_attribute_filter(lang)
        )
        for seed in range(5):
            random.seed(seed)
            ref = tt._apply_expression_attributes(
                [meta[1]], [("a bicycle", [])], alts, attrs, cfg
            )
            random.seed(seed)
            got = fs.apply_expression_attributes(
                [meta[1]], [("a bicycle", [])], alts, attrs, cfg
            )
            assert ref == got
