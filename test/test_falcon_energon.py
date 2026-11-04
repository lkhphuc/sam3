import random

import numpy as np
import pytest
import torch

from falcon_fixtures import raw_sample, rect_mask, rle
from sam3.train.data.falcon_energon import (
    collate_samples,
    Sam3FalconTaskEncoder,
    unpack_mask_bits,
)
from sam3.train.data.falcon_schema import unpack_masks


def _encoder(**kw):
    args = dict(
        resolution=64,
        resize_scales=[64],
        random_pad=False,
        max_queries_per_sample=64,
        max_queries_per_batch=64,
        max_instances_per_batch=64,
        max_instances_per_query=50,
        max_images_per_batch=8,
        log_stats_every=0,
    )
    args.update(kw)
    return Sam3FalconTaskEncoder(**args)


def _encode(enc, sample):
    # The undecorated method: @stateless(restore_seeds=True) needs an energon
    # worker context to seed from.
    fn = type(enc).encode_sample
    return list(getattr(fn, "__wrapped__", fn)(enc, sample))


def test_encode_sample_masks_boxes_and_queries():
    random.seed(0)
    enc = _encoder(resolution=1008, resize_scales=[1008])
    sample, masks = raw_sample(h=300, w=400)
    (s,) = _encode(enc, sample)
    assert s.image.shape == (3, 1008, 1008) and s.image.dtype == np.uint8
    by_text = dict(zip(s.texts, s.num_instances.tolist()))
    assert by_text["dog"] == 2 and by_text["bicycle"] == 0 and by_text["boat"] == 0
    assert len(s.texts) == 5 - 1  # dog, cat (key or alternative), 2 negatives
    # The dog masks: nearest-resized rectangles, and their tight boxes.
    starts = np.cumsum(s.num_instances) - s.num_instances
    i = s.texts.index("dog")
    dog_bits = s.mask_bits[starts[i] : starts[i] + 2]
    dog = unpack_masks(dog_bits, 1008)
    sy, sx = 1008 / 300, 1008 / 400
    # first dog: rows 10..60, cols 20..100 of 300x400
    ys, xs = np.nonzero(dog[0])
    assert abs(ys.min() - 10 * sy) <= 2 and abs(ys.max() + 1 - 60 * sy) <= 2
    assert abs(xs.min() - 20 * sx) <= 2 and abs(xs.max() + 1 - 100 * sx) <= 2
    cx, cy, w, h = s.boxes[starts[i]]
    np.testing.assert_allclose(
        [cx, cy, w, h],
        [
            (xs.min() + xs.max() + 1) / 2 / 1008,
            (ys.min() + ys.max() + 1) / 2 / 1008,
            (xs.max() + 1 - xs.min()) / 1008,
            (ys.max() + 1 - ys.min()) / 1008,
        ],
        rtol=1e-6,
    )
    assert s.original_size == (300, 400)


def test_random_scale_and_pad_moves_masks_with_image():
    random.seed(3)
    enc = _encoder(resolution=128, resize_scales=[64], random_pad=True)
    sample, _ = raw_sample(h=64, w=64)
    sample.metadata = [
        {
            "task": "referring_segmentation",
            "box": [rle(rect_mask(64, 64, 10, 20, 30, 50))],
        }
    ]
    (s,) = _encode(enc, sample)
    content = s.image.any(axis=0)
    ys, xs = np.nonzero(content)
    off_y, off_x = ys.min(), xs.min()
    m = unpack_masks(s.mask_bits, 128)[0]
    mys, mxs = np.nonzero(m)
    assert (mys.min(), mxs.min()) == (off_y + 10, off_x + 30)
    assert s.boxes[0][2] == pytest.approx(20 / 128)


def test_too_many_instances_drops_query_not_subsample():
    enc = _encoder(max_instances_per_query=1)
    sample, _ = raw_sample()
    (s,) = _encode(enc, sample)
    assert "dog" not in s.texts  # 2 instances > 1
    assert enc.stats["query_dropped_too_many_instances"] == 1


def test_vanished_instance_drops_whole_query_if_empty():
    enc = _encoder(resolution=8, resize_scales=[8])
    sample, _ = raw_sample(h=1000, w=1000)
    tiny = rect_mask(1000, 1000, 500, 501, 500, 501)  # 1 px: gone at 8x8 nearest
    sample.metadata = [
        {
            "task": "referring_segmentation",
            "speck": [rle(tiny)],
            "big": [rle(rect_mask(1000, 1000, 0, 900, 0, 900))],
        },
    ]
    (s,) = _encode(enc, sample)
    assert s.texts == ["big"]
    assert enc.stats["query_dropped_all_instances_vanished"] == 1


def test_split_and_pack_budgets():
    enc = _encoder(
        max_queries_per_sample=2, max_queries_per_batch=5, max_images_per_batch=8
    )
    splits = []
    for k in range(3):
        sample, _ = raw_sample(key=f"src/{k}")
        splits += _encode(enc, sample)
    # 4 queries per image -> 2 splits of 2 queries each.
    assert len(splits) == 6 and all(s.num_queries == 2 for s in splits)
    assert splits[1].__key__.endswith("#1")
    groups = enc.select_samples_to_pack(splits)
    for g in groups:
        assert sum(s.num_queries for s in g) <= 5
    assert sum(len(g) for g in groups) == 6


def test_skip_when_nothing_left():
    from sam3.train.data.falcon_energon import SkipSample

    enc = _encoder()
    sample, _ = raw_sample()
    sample.metadata = [
        {"task": "object detection", "dog": [{"x1": 1, "y1": 1, "x2": 2, "y2": 2}]}
    ]
    with pytest.raises(SkipSample):
        _encode(enc, sample)


def test_unpack_mask_bits_matches_numpy():
    rng = np.random.default_rng(0)
    m = (rng.random((4, 9, 1008)) > 0.5).astype(np.uint8)
    bits = torch.from_numpy(np.packbits(m, axis=-1))
    np.testing.assert_array_equal(unpack_mask_bits(bits, 1008).numpy(), m.astype(bool))


def _reference_batch(samples, mean=(0.5,) * 3, std=(0.5,) * 3):
    """The same data through sam3's own Datapoint -> collate_fn_api path."""
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image,
        InferenceMetadata,
        Object,
    )

    datapoints = []
    for s in samples:
        R = s.image.shape[-1]
        img = torch.from_numpy(s.image).float() / 255.0
        img = (img - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
        masks = unpack_masks(s.mask_bits, R).astype(bool)
        objects = [
            Object(
                bbox=torch.from_numpy(s.boxes[i]),
                area=0.0,
                segment=torch.from_numpy(masks[i]),
            )
            for i in range(len(s.boxes))
        ]
        queries, start = [], 0
        for t, n in zip(s.texts, s.num_instances.tolist()):
            queries.append(
                FindQueryLoaded(
                    query_text=t,
                    image_id=0,
                    object_ids_output=list(range(start, start + n)),
                    is_exhaustive=True,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=-1,
                        original_image_id=-1,
                        original_category_id=-1,
                        original_size=s.original_size,
                        object_id=-1,
                        frame_index=-1,
                    ),
                )
            )
            start += n
        datapoints.append(
            Datapoint(
                find_queries=queries,
                images=[Image(data=img, objects=objects, size=(R, R))],
            )
        )
    return collate_fn_api(datapoints, dict_key="all", with_seg_masks=True, repeats=1)[
        "all"
    ]


def test_batch_equals_sam3_collator():
    random.seed(1)
    enc = _encoder(resolution=96, resize_scales=[64, 96], random_pad=True)
    samples = []
    for k in range(3):
        sample, _ = raw_sample(key=f"src/{k}")
        samples += _encode(enc, sample)
    raw = (
        collate_samples(samples).pin_memory()
        if torch.cuda.is_available()
        else collate_samples(samples)
    )
    got = raw.to_batched_datapoint(torch.device("cpu"))
    ref = _reference_batch(samples)

    torch.testing.assert_close(got.img_batch, ref.img_batch)
    assert got.find_text_batch == ref.find_text_batch
    gi, ri = got.find_inputs[0], ref.find_inputs[0]
    for name in (
        "img_ids",
        "text_ids",
        "input_boxes",
        "input_boxes_mask",
        "input_boxes_label",
        "input_points",
        "input_points_mask",
    ):
        a, b = getattr(gi, name), getattr(ri, name)
        assert a.shape == b.shape and a.dtype == b.dtype, name
        torch.testing.assert_close(a, b, msg=name)
    assert gi.object_ids == ri.object_ids
    gt, rt = got.find_targets[0], ref.find_targets[0]
    for name in (
        "num_boxes",
        "boxes",
        "boxes_padded",
        "repeated_boxes",
        "segments",
        "is_valid_segment",
        "is_exhaustive",
        "object_ids",
        "object_ids_padded",
    ):
        a, b = getattr(gt, name), getattr(rt, name)
        assert a.shape == b.shape and a.dtype == b.dtype, name
        torch.testing.assert_close(a, b, msg=name)
    gm, rm = got.find_metadatas[0], ref.find_metadatas[0]
    for name in (
        "original_size",
        "coco_image_id",
        "original_image_id",
        "object_id",
        "frame_index",
    ):
        torch.testing.assert_close(getattr(gm, name), getattr(rm, name), msg=name)
