"""The SAM3 task encoder inside a real (upstream) energon pipeline.

Upstream energon has no NpyTxtJsonSample (a tiiuae fork type), so the test
dataset uses a stand-in sample type with the same `metadata` JSON and a single
`image`; the encoder reads either `images[0]` or `image`. Covers the generator
encode_sample, packing, batching, worker processes and save/restore.
"""

import io
import json
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from typing import Any

import pytest
import torch

pytest.importorskip("megatron.energon")
from megatron.energon import Sample  # noqa: E402

from falcon_fixtures import image_bytes, perception_metadata  # noqa: E402
from sam3.train.data.falcon_energon import (  # noqa: E402
    Sam3EnergonLoader,
    Sam3FalconBatch,
    Sam3FalconTaskEncoder,
)


@dataclass
class FalconLikeSample(Sample):
    image: Any
    metadata: Any


def _write_dataset(root, n=12):
    root.mkdir()
    with tarfile.open(root / "shard_000.tar", "w") as tar:
        for i in range(n):
            meta, _ = perception_metadata(h=120 + i, w=160)
            for ext, data in (
                ("png", image_bytes(120 + i, 160, seed=i)),
                ("json", json.dumps(meta).encode()),
            ):
                info = tarfile.TarInfo(f"{i:06d}.{ext}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "megatron.energon.cli.main",
            "prepare",
            str(root),
            "--non-interactive",
            "--split-ratio",
            "1,0,0",
            "--skip-dataset-yaml",
            "--no-progress",
        ],
        check=True,
        capture_output=True,
    )
    (root / ".nv-meta" / "dataset.yaml").write_text(
        "sample_type:\n  __module__: test_falcon_energon_e2e\n  __class__: FalconLikeSample\n"
        "field_map:\n  image: png\n  metadata: json\n"
    )


def _loader(path, num_workers):
    enc = Sam3FalconTaskEncoder(
        resolution=64,
        resize_scales=[48, 64],
        max_queries_per_sample=3,
        max_queries_per_batch=8,
        max_instances_per_batch=40,
        max_images_per_batch=4,
        max_instances_per_query=10,
        log_stats_every=0,
    )
    return Sam3EnergonLoader(
        dataset_path=str(path),
        task_encoder=enc,
        num_workers=num_workers,
        shuffle_buffer_size=4,
        packing_buffer_size=6,
        max_samples_per_sequence=4,
        watchdog_timeout_seconds=None,
    )


@pytest.mark.parametrize("num_workers", [0, 2])
def test_energon_pipeline_and_resume(tmp_path, monkeypatch, num_workers):
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).parent))
    try:
        _write_dataset(tmp_path / "ds")
    except subprocess.CalledProcessError as e:
        pytest.skip(f"energon prepare failed: {e.stderr.decode()[-500:]}")
    loader = _loader(tmp_path / "ds", num_workers)
    it = iter(loader)
    batches = [next(it) for _ in range(4)]
    for b in batches:
        assert isinstance(b, Sam3FalconBatch)
        assert b.num_queries <= 8 and b.images.shape[0] <= 4 and b.boxes.shape[0] <= 40
        assert b.images.shape[1:] == (3, 64, 64) and b.mask_bits.shape[1:] == (64, 8)
        assert int(b.num_boxes.sum()) == b.boxes.shape[0] == b.mask_bits.shape[0]
        bd = b.to_batched_datapoint(torch.device("cpu"))
        assert bd.find_targets[0].segments.shape == (b.boxes.shape[0], 64, 64)
    state = loader.state_dict()
    expected = [next(it).__key__ for _ in range(3)]
    del it, loader

    resumed = _loader(tmp_path / "ds", num_workers)
    resumed.load_state_dict(state)
    it2 = iter(resumed)
    got = [next(it2).__key__ for _ in range(3)]
    assert got == expected
