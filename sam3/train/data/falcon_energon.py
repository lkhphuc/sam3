# pyre-unsafe
"""Energon loader for training SAM3 on the Falcon Perception v1.5 segmentation data.

Pipeline (all CPU work happens in the loader workers):

    NpyTxtJsonSample (image bytes + perception JSON)
      -> encode_sample: parse the referring-segmentation queries and negatives,
         resize the image to a random square size (SAM3 train recipe),
         rasterise every RLE straight onto that grid, pad to `resolution`,
         bit-pack the masks, tight boxes from the masks.
         An image with more than `max_queries_per_sample` queries is split into
         several samples of the same image (queries are independent in SAM3).
      -> select_samples_to_pack / pack_selected_samples: first-fit packing of
         images under a per-step budget of queries, instances and images.
      -> batch: one Sam3FalconBatch per rank step (uint8 images, packed masks).
    Trainer: Sam3FalconBatch.to_batched_datapoint(device) unpacks the masks and
    normalises the images on the GPU and returns the `BatchedDatapoint` that
    `Sam3Image.forward` and `Sam3LossWrapper` consume, identical to what
    `collate_fn_api` builds from the same data (test/test_falcon_energon.py).

Data semantics:
  * Only `referring_segmentation` objects give positives; `negative_expressions`
    give the negatives (0 targets). Box-only DET, pointing, exemplar and caption
    objects are ignored.
  * All queries are exhaustive. A query with more than `max_instances_per_query`
    instances (the decoder's 200 object queries) is dropped, never subsampled:
    subsampling would train the left-out instances as background.
  * Instances that vanish at the sampled resolution (0 pixels) are removed; a
    positive query left with no instance is dropped (it is not a negative).
"""

import io
import logging
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image as PILImage

from .falcon_schema import (
    AttributeFilter,
    decode_masks_resized,
    pack_masks,
    parse_attribute_filter,
    parse_segmentation_queries,
    SegQuery,
)

logger = logging.getLogger(__name__)

try:  # The loader runs inside energon; the encoder logic is testable without it.
    from megatron.energon import DefaultTaskEncoder, SkipSample, stateless
except ImportError:  # pragma: no cover - energon is a training-time dependency
    DefaultTaskEncoder = object

    class SkipSample(Exception):
        pass

    def stateless(fn=None, **_kwargs):
        return fn if fn is not None else (lambda f: f)


# ---------------------------------------------------------------------------
# Encoded types
# ---------------------------------------------------------------------------


@dataclass
class Sam3FalconSample:
    """One image (or one split of an image) with its find queries."""

    __key__: str
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    image: np.ndarray  # uint8 [3, R, R], content in the top-left-offset box, rest 0
    texts: List[str]  # one per query
    num_instances: np.ndarray  # int64 [Q]
    boxes: np.ndarray  # float32 [N, 4] normalised cxcywh in the R x R canvas
    mask_bits: np.ndarray  # uint8 [N, R, ceil(R/8)], np.packbits along W
    source: str = ""
    original_size: Tuple[int, int] = (0, 0)  # (h, w) of the stored image

    @property
    def num_queries(self) -> int:
        return len(self.texts)

    @property
    def total_instances(self) -> int:
        return int(self.num_instances.sum())


@dataclass
class Sam3FalconPack:
    """Images packed into one rank step by select_samples_to_pack."""

    __key__: str
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    samples: List[Sam3FalconSample]


@dataclass
class Sam3FalconBatch:
    """One rank step, as produced in the loader workers.

    Everything is a CPU tensor, so the DataLoader can pin it; pinning needs the
    `pin_memory` method below (torch's pin utility does not recurse into a
    dataclass).
    """

    __key__: List[str]
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    images: torch.Tensor  # uint8 [B, 3, R, R]
    texts: List[str]  # unique query texts of the batch (find_text_batch)
    img_ids: torch.Tensor  # int64 [Q] image of each query
    text_ids: torch.Tensor  # int64 [Q] index into texts
    num_boxes: torch.Tensor  # int64 [Q]
    boxes: torch.Tensor  # float32 [N, 4] cxcywh, normalised
    mask_bits: torch.Tensor  # uint8 [N, R, ceil(R/8)]
    original_sizes: torch.Tensor  # int64 [Q, 2]
    sources: List[str]  # per image

    @property
    def resolution(self) -> int:
        return int(self.images.shape[-1])

    @property
    def num_queries(self) -> int:
        return int(self.img_ids.shape[0])

    def pin_memory(self) -> "Sam3FalconBatch":
        for name in (
            "images",
            "img_ids",
            "text_ids",
            "num_boxes",
            "boxes",
            "mask_bits",
            "original_sizes",
        ):
            setattr(self, name, getattr(self, name).pin_memory())
        return self

    def to_batched_datapoint(
        self,
        device: torch.device,
        *,
        norm_mean: Sequence[float] = (0.5, 0.5, 0.5),
        norm_std: Sequence[float] = (0.5, 0.5, 0.5),
        with_masks: bool = True,
    ):
        """The SAM3 `BatchedDatapoint` on `device` (one find stage, no geometric
        prompts), equal to `collate_fn_api(..., with_seg_masks=True, repeats=1)`
        on the same data with ToTensor + Normalize applied."""
        from sam3.model.data_misc import (
            BatchedDatapoint,
            BatchedFindTarget,
            BatchedInferenceMetadata,
            FindStage,
        )

        nb = lambda t: t.to(device, non_blocking=True)  # noqa: E731
        num_queries = self.num_queries
        mean = torch.tensor(norm_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(norm_std, device=device).view(1, 3, 1, 1)
        img = (nb(self.images).float().div_(255.0) - mean) / std

        num_boxes = nb(self.num_boxes)
        boxes = nb(self.boxes)
        max_n = int(self.num_boxes.max()) if num_queries else 0  # CPU tensor: no sync
        boxes_padded = boxes.new_zeros(num_queries, max_n, 4)
        object_ids_padded = torch.full(
            (num_queries, max_n), -1, dtype=torch.long, device=device
        )
        # Instance index within its query, and the query of each instance.
        counts = self.num_boxes
        query_of_instance = torch.repeat_interleave(torch.arange(num_queries), counts)
        starts = torch.cumsum(counts, 0) - counts
        within = torch.arange(int(counts.sum())) - starts[query_of_instance]
        boxes_padded[nb(query_of_instance), nb(within)] = boxes
        # collate_fn_api's object ids index the objects of the query's image;
        # here every instance is its own object, numbered per image.
        object_ids = _object_ids_per_image(self.img_ids, counts)
        object_ids_padded[nb(query_of_instance), nb(within)] = nb(object_ids)

        segments = None
        is_valid = None
        if with_masks:
            segments = unpack_mask_bits(nb(self.mask_bits), self.resolution)
            is_valid = torch.ones(segments.shape[0], dtype=torch.bool, device=device)

        object_ids_list = _split_list(object_ids.tolist(), counts.tolist())
        find_input = FindStage(
            img_ids=nb(self.img_ids),
            text_ids=nb(self.text_ids),
            input_boxes=torch.zeros(0, num_queries, 4, device=device),
            input_boxes_mask=torch.ones(
                num_queries, 0, dtype=torch.bool, device=device
            ),
            input_boxes_label=torch.zeros(
                0, num_queries, dtype=torch.long, device=device
            ),
            input_points=torch.zeros(num_queries, 0, 257, device=device),
            input_points_mask=torch.ones(
                num_queries, 0, dtype=torch.bool, device=device
            ),
            object_ids=object_ids_list,
        )
        find_target = BatchedFindTarget(
            num_boxes=num_boxes,
            boxes=boxes,
            boxes_padded=boxes_padded,
            repeated_boxes=boxes,
            segments=segments,
            semantic_segments=torch.zeros(0, dtype=torch.bool, device=device),
            is_valid_segment=is_valid,
            is_exhaustive=torch.ones(num_queries, dtype=torch.bool, device=device),
            object_ids=nb(object_ids),
            object_ids_padded=object_ids_padded,
        )
        minus_one = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        metadata = BatchedInferenceMetadata(
            coco_image_id=minus_one,
            original_image_id=minus_one.clone(),
            original_category_id=minus_one.to(torch.int),
            original_size=nb(self.original_sizes),
            object_id=minus_one.clone(),
            frame_index=minus_one.clone(),
            is_conditioning_only=[False] * num_queries,
        )
        return BatchedDatapoint(
            img_batch=img,
            find_text_batch=list(self.texts),
            find_inputs=[find_input],
            find_targets=[find_target],
            find_metadatas=[metadata],
            raw_images=None,
        )


def _split_list(values: List[int], counts: List[int]) -> List[List[int]]:
    out, i = [], 0
    for c in counts:
        out.append(values[i : i + c])
        i += c
    return out


def _object_ids_per_image(img_ids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Object id of each instance: its index among the instances of its image."""
    ids = []
    seen: Dict[int, int] = {}
    for img, c in zip(img_ids.tolist(), counts.tolist()):
        start = seen.get(img, 0)
        ids.extend(range(start, start + c))
        seen[img] = start + c
    return torch.tensor(ids, dtype=torch.long)


_BIT_SHIFTS = torch.arange(7, -1, -1, dtype=torch.uint8)


def unpack_mask_bits(bits: torch.Tensor, width: int) -> torch.Tensor:
    """[N, H, W/8] uint8 (np.packbits, MSB first) -> [N, H, width] bool, on bits' device."""
    shifts = _BIT_SHIFTS.to(bits.device)
    out = (bits.unsqueeze(-1) >> shifts).bitwise_and_(1).bool()
    return out.flatten(-2)[..., :width]


# ---------------------------------------------------------------------------
# Task encoder
# ---------------------------------------------------------------------------


def _decode_image(raw: Any) -> PILImage.Image:
    if isinstance(raw, PILImage.Image):
        img = raw
    elif isinstance(raw, (bytes, bytearray, np.bytes_)):
        img = PILImage.open(io.BytesIO(raw))
    elif isinstance(raw, np.ndarray):
        if raw.dtype == np.uint8 and raw.ndim == 1:  # encoded bytes as an array
            img = PILImage.open(io.BytesIO(raw.tobytes()))
        else:
            img = PILImage.fromarray(raw)
    elif isinstance(
        raw, torch.Tensor
    ):  # energon "torchrgb"-style decode, CHW float/uint8
        arr = raw.detach().cpu()
        if arr.dtype != torch.uint8:
            arr = (arr.clamp(0, 1) * 255).round().to(torch.uint8)
        img = PILImage.fromarray(arr.permute(1, 2, 0).numpy())
    else:
        img = PILImage.open(raw)
    # No EXIF transpose: the annotations refer to the stored pixel grid, as in
    # the Falcon loader (dataset_utils.process_images).
    return img.convert("RGB")


def default_resize_scales(
    resolution: int, min_size: int = 480, stride: int = 32
) -> List[int]:
    """sam3.train.transforms.basic.get_random_resize_scales(resolution, min_size, rounded=False)."""
    min_size = int(stride * math.ceil(min_size / stride))
    return list(range(min_size, resolution + 1, stride))


class Sam3FalconTaskEncoder(DefaultTaskEncoder):
    """Falcon Perception v1.5 perception samples -> SAM3 training batches.

    Per-source subflavors read by this encoder (all optional):
        level, lang       expression filters, as in the Falcon loader (default: no filter)
        max_instances_per_query   overrides the encoder default
        max_negatives_per_image   overrides the encoder default
    """

    def __init__(
        self,
        *,
        resolution: int = 1008,
        resize_scales: Optional[Sequence[int]] = None,
        random_pad: bool = True,
        max_instances_per_query: int = 200,
        max_negatives_per_image: int = -1,
        max_queries_per_sample: int = 64,
        max_queries_per_batch: int = 256,
        max_instances_per_batch: int = 4000,
        max_images_per_batch: int = 16,
        strip_articles: bool = True,
        log_stats_every: int = 2000,
    ):
        super().__init__()
        if resolution % 8:
            raise ValueError(f"resolution must be a multiple of 8, got {resolution}")
        self.resolution = resolution
        self.resize_scales = (
            list(resize_scales) if resize_scales else default_resize_scales(resolution)
        )
        if max(self.resize_scales) > resolution:
            raise ValueError("resize_scales must not exceed resolution")
        self.random_pad = random_pad
        self.max_instances_per_query = max_instances_per_query
        self.max_negatives_per_image = max_negatives_per_image
        self.max_queries_per_sample = max_queries_per_sample
        self.max_queries_per_batch = max_queries_per_batch
        self.max_instances_per_batch = max_instances_per_batch
        self.max_images_per_batch = max_images_per_batch
        if max_queries_per_sample > max_queries_per_batch:
            raise ValueError("max_queries_per_sample must be <= max_queries_per_batch")
        if max_instances_per_query > max_instances_per_batch:
            raise ValueError(
                "max_instances_per_query must be <= max_instances_per_batch"
            )
        self.strip_articles = strip_articles
        self.log_stats_every = log_stats_every
        self.stats: Counter = Counter()

    # -- per sample ---------------------------------------------------------

    def _bump(self, key: str, n: int = 1) -> None:
        self.stats[key] += n
        if (
            self.log_stats_every
            and key == "samples_in"
            and self.stats[key] % self.log_stats_every == 0
        ):
            logger.info(f"Sam3FalconTaskEncoder worker stats: {dict(self.stats)}")

    def select_queries(
        self, queries: List[SegQuery], subflavors: Dict[str, Any]
    ) -> List[SegQuery]:
        max_inst = int(
            subflavors.get("max_instances_per_query", self.max_instances_per_query)
        )
        max_neg = int(
            subflavors.get("max_negatives_per_image", self.max_negatives_per_image)
        )
        positives = []
        for q in queries:
            if q.is_negative:
                continue
            if len(q.rles) > max_inst:
                self._bump("query_dropped_too_many_instances")
                continue
            positives.append(q)
        negatives = [q for q in queries if q.is_negative]
        if 0 <= max_neg < len(negatives):
            self._bump("negatives_dropped_cap", len(negatives) - max_neg)
            negatives = random.sample(negatives, max_neg)
        selected = positives + negatives
        random.shuffle(selected)
        return selected

    def build_samples(
        self,
        key: str,
        restore_key: tuple,
        raw_image: Any,
        metadata: Any,
        subflavors: Optional[Dict[str, Any]],
    ) -> List[Sam3FalconSample]:
        """The encoded split(s) of one raw sample (an empty list = skip)."""
        subflavors = subflavors or {}
        self._bump("samples_in")
        attribute_filter = AttributeFilter(
            level=parse_attribute_filter(subflavors.get("level")),
            lang=parse_attribute_filter(subflavors.get("lang")),
        )
        queries, parse_stats = parse_segmentation_queries(
            metadata, attribute_filter, strip_articles=self.strip_articles
        )
        for k, v in parse_stats.items():
            if v:
                self._bump(k, v)
        queries = self.select_queries(queries, subflavors)
        if not queries:
            self._bump("sample_skipped_no_query")
            return []

        image = _decode_image(raw_image)
        orig_w, orig_h = image.size
        size = random.choice(self.resize_scales)
        R = self.resolution
        # Square resize (sam3 RandomResizeAPI square=True; PIL bilinear), then
        # pad to R x R at a random offset (PadToSizeAPI) or top-left.
        image = image.resize((size, size), PILImage.BILINEAR)
        pad = R - size
        off_x = random.randint(0, pad) if self.random_pad else 0
        off_y = random.randint(0, pad) if self.random_pad else 0
        canvas = np.zeros((3, R, R), np.uint8)
        canvas[:, off_y : off_y + size, off_x : off_x + size] = np.asarray(
            image, np.uint8
        ).transpose(2, 0, 1)

        kept_texts, kept_counts, kept_masks = [], [], []
        for q in queries:
            if q.is_negative:
                kept_texts.append(q.text)
                kept_counts.append(0)
                continue
            masks = [m for m in decode_masks_resized(q.rles, size, size) if m.any()]
            if len(masks) < len(q.rles):
                self._bump("instances_vanished_at_scale", len(q.rles) - len(masks))
            if not masks:
                self._bump("query_dropped_all_instances_vanished")
                continue
            kept_texts.append(q.text)
            kept_counts.append(len(masks))
            kept_masks.extend(masks)
        if not kept_texts:
            self._bump("sample_skipped_no_query")
            return []

        n = len(kept_masks)
        boxes = np.zeros((n, 4), np.float32)
        bits = np.zeros((n, R, (R + 7) // 8), np.uint8)
        for i, m in enumerate(kept_masks):
            ys, xs = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
            x0, x1 = xs[0] + off_x, xs[-1] + 1 + off_x
            y0, y1 = ys[0] + off_y, ys[-1] + 1 + off_y
            boxes[i] = (
                (x0 + x1) / 2 / R,
                (y0 + y1) / 2 / R,
                (x1 - x0) / R,
                (y1 - y0) / R,
            )
            full = np.zeros((R, R), np.uint8)
            full[off_y : off_y + size, off_x : off_x + size] = m
            bits[i] = pack_masks(full)

        source = str(subflavors.get("source", "")) or key.split("/")[0]
        counts = np.asarray(kept_counts, np.int64)
        out = []
        chunk = self.max_queries_per_sample
        for split, start in enumerate(range(0, len(kept_texts), chunk)):
            stop = min(start + chunk, len(kept_texts))
            inst0 = int(counts[:start].sum())
            inst1 = int(counts[:stop].sum())
            out.append(
                Sam3FalconSample(
                    __key__=key if split == 0 else f"{key}#{split}",
                    __restore_key__=restore_key,
                    image=canvas,
                    texts=kept_texts[start:stop],
                    num_instances=counts[start:stop],
                    boxes=boxes[inst0:inst1],
                    mask_bits=bits[inst0:inst1],
                    source=source,
                    original_size=(orig_h, orig_w),
                )
            )
        if len(out) > 1:
            self._bump("samples_split", len(out) - 1)
        self._bump("queries_out", len(kept_texts))
        self._bump("instances_out", n)
        return out

    @stateless(restore_seeds=True)
    def encode_sample(self, sample) -> Iterator[Sam3FalconSample]:
        """Always a generator (energon restore keys depend on it; see
        torchtitan-internal falcon_taskencoder.encode_sample)."""
        images = getattr(sample, "images", None)
        if images is None:
            images = [getattr(sample, "image")]
        if len(images) != 1:
            # Multi-image samples are exemplar data, not used for SAM3.
            self._bump("sample_skipped_multi_image")
            raise SkipSample()
        encoded = self.build_samples(
            sample.__key__,
            getattr(sample, "__restore_key__", ()),
            images[0],
            sample.metadata,
            getattr(sample, "__subflavors__", None),
        )
        if not encoded:
            raise SkipSample()
        yield from encoded

    # -- packing: one rank step under a query / instance / image budget ------

    def _fits(self, q: int, n: int, k: int) -> bool:
        return (
            q <= self.max_queries_per_batch
            and n <= self.max_instances_per_batch
            and k <= self.max_images_per_batch
        )

    def select_samples_to_pack(
        self, samples: List[Sam3FalconSample]
    ) -> List[List[Sam3FalconSample]]:
        """First-fit in buffer order. A sample over the instance budget on its
        own (only possible when the budgets are inconsistent) is dropped."""
        groups: List[List[Sam3FalconSample]] = []
        totals: List[List[int]] = []  # queries, instances, images
        for s in samples:
            q, n = s.num_queries, s.total_instances
            if not self._fits(q, n, 1):
                self._bump("sample_dropped_over_batch_budget")
                continue
            for g, t in zip(groups, totals):
                if self._fits(t[0] + q, t[1] + n, t[2] + 1):
                    g.append(s)
                    t[0] += q
                    t[1] += n
                    t[2] += 1
                    break
            else:
                groups.append([s])
                totals.append([q, n, 1])
        return groups

    @stateless
    def pack_selected_samples(self, samples: List[Sam3FalconSample]) -> Sam3FalconPack:
        return Sam3FalconPack(
            __key__=",".join(s.__key__ for s in samples),
            __restore_key__=(),
            samples=list(samples),
        )

    # -- batching -----------------------------------------------------------

    def batch(
        self, samples: List[Union[Sam3FalconSample, Sam3FalconPack]]
    ) -> Sam3FalconBatch:
        flat: List[Sam3FalconSample] = []
        for s in samples:
            flat.extend(s.samples if isinstance(s, Sam3FalconPack) else [s])
        return collate_samples(flat)

    def encode_batch(self, batch: Sam3FalconBatch) -> Sam3FalconBatch:
        return batch


def collate_samples(samples: List[Sam3FalconSample]) -> Sam3FalconBatch:
    texts: List[str] = []
    text_index: Dict[str, int] = {}
    img_ids, text_ids, counts, sizes = [], [], [], []
    for i, s in enumerate(samples):
        for t, c in zip(s.texts, s.num_instances.tolist()):
            if t not in text_index:
                text_index[t] = len(texts)
                texts.append(t)
            img_ids.append(i)
            text_ids.append(text_index[t])
            counts.append(c)
            sizes.append(s.original_size)
    R = samples[0].image.shape[-1]
    boxes = [s.boxes for s in samples if len(s.boxes)]
    bits = [s.mask_bits for s in samples if len(s.mask_bits)]
    return Sam3FalconBatch(
        __key__=[s.__key__ for s in samples],
        __restore_key__=(),
        images=torch.from_numpy(np.stack([s.image for s in samples])),
        texts=texts,
        img_ids=torch.tensor(img_ids, dtype=torch.long),
        text_ids=torch.tensor(text_ids, dtype=torch.long),
        num_boxes=torch.tensor(counts, dtype=torch.long),
        boxes=torch.from_numpy(np.concatenate(boxes)) if boxes else torch.zeros(0, 4),
        mask_bits=(
            torch.from_numpy(np.concatenate(bits))
            if bits
            else torch.zeros(0, R, (R + 7) // 8, dtype=torch.uint8)
        ),
        original_sizes=torch.tensor(sizes, dtype=torch.long).view(-1, 2),
        sources=[s.source for s in samples],
    )


# ---------------------------------------------------------------------------
# Loader construction
# ---------------------------------------------------------------------------


def disable_numpy_hugepages() -> None:
    """Copied from torchtitan-internal energon_datasets.py: stop numpy from
    madvise(MADV_HUGEPAGE)-ing large arrays. On the B200 nodes THP compaction
    stalled loader workers for minutes once RAM filled with page cache."""
    try:
        from numpy._core.multiarray import _set_madvise_hugepage
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import _set_madvise_hugepage
    _set_madvise_hugepage(False)


def raise_open_file_limit() -> None:
    """Copied from torchtitan-internal: workers hold one fd per shared-memory
    tensor in flight; the default soft limit (1024) dropped mask-heavy batches
    and hung the rank."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and (
        hard == resource.RLIM_INFINITY or soft < hard
    ):
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        logger.info(f"Raised the open-file limit from {soft} to {hard}")


class Sam3EnergonLoader:
    """Savable energon loader for one data-parallel rank (Hydra target).

    `state_dict()` / `load_state_dict()` save and restore this rank's position;
    resuming needs the same world size, num_workers and blend (append-only).
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        task_encoder: Sam3FalconTaskEncoder,
        num_workers: int = 8,
        shuffle_buffer_size: int = 64,
        packing_buffer_size: Optional[int] = 128,
        max_samples_per_sequence: Optional[int] = 16,
        metadata_folder_name: str = ".nv-meta",
        watchdog_timeout_seconds: Optional[float] = 300,
        seed_offset: int = 0,
    ):
        import inspect

        import torch.distributed as dist
        from megatron.energon import get_train_dataset, WorkerConfig
        from megatron.energon.savable_loader import SavableDataLoader

        disable_numpy_hugepages()
        raise_open_file_limit()
        if metadata_folder_name != ".nv-meta":
            from megatron.energon.flavors.webdataset.config import set_main_folder_name

            set_main_folder_name(metadata_folder_name)
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = rank
        worker_config = WorkerConfig(
            rank=rank,
            world_size=world,
            num_workers=num_workers,
            seed_offset=seed_offset,
        )
        kwargs = dict(
            batch_size=1 if packing_buffer_size else task_encoder.max_images_per_batch,
            packing_buffer_size=packing_buffer_size,
            shuffle_buffer_size=shuffle_buffer_size,
            max_samples_per_sequence=max_samples_per_sequence,
            worker_config=worker_config,
            task_encoder=task_encoder,
            repeat=True,
        )
        params = inspect.signature(get_train_dataset).parameters
        # Same decode as the Falcon loader; _decode_image also accepts raw bytes.
        kwargs["image_decode"] = "pilrgb"
        # The tiiuae fork takes the worker watchdog on get_train_dataset, upstream
        # energon on SavableDataLoader.
        loader_kwargs = {}
        if watchdog_timeout_seconds is not None:
            if "watchdog_timeout_seconds" in params:
                kwargs["watchdog_timeout_seconds"] = watchdog_timeout_seconds
            elif (
                "watchdog_timeout_seconds"
                in inspect.signature(SavableDataLoader).parameters
            ):
                loader_kwargs["watchdog_timeout_seconds"] = watchdog_timeout_seconds
        if packing_buffer_size is None:
            logger.warning(
                "packing_buffer_size=None: batches hold a fixed number of images and "
                "max_queries_per_batch / max_instances_per_batch are not enforced"
            )
        dataset = get_train_dataset(dataset_path, **kwargs)
        self.loader = SavableDataLoader(dataset, **loader_kwargs)

    def __iter__(self):
        return iter(self.loader)

    def state_dict(self) -> dict:
        return {f"dp_rank{self.rank}": self.loader.save_state_rank()}

    def load_state_dict(self, state: dict) -> None:
        self.loader.restore_state_rank(state[f"dp_rank{self.rank}"])
