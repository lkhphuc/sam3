"""Eval wiring for StreamingTrainer checkpoints (CPU): held-out set conversion,
the SA-Co loader on it through the eval config, cgF1 slicing, eval model build."""

import io
import json
import os
import subprocess
import sys
import tarfile

import numpy as np
import pytest
import torch

from falcon_fixtures import image_bytes, perception_metadata

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts", "falcon_v15")


def _dataset(root, n=4):
    """An energon-style source with a train and a val tar (npy image + json)."""
    (root / ".nv-meta").mkdir(parents=True)
    for split in ("train", "val"):
        with tarfile.open(root / f"{split}_000.tar", "w") as t:
            for i in range(n):
                meta, _ = perception_metadata()
                buf = io.BytesIO()
                np.save(
                    buf,
                    np.array([image_bytes(300, 400, seed=i)], dtype=object),
                    allow_pickle=True,
                )
                for ext, data in (
                    ("npy", buf.getvalue()),
                    ("json", json.dumps(meta).encode()),
                ):
                    ti = tarfile.TarInfo(f"{split}{i:05d}.{ext}")
                    ti.size = len(data)
                    t.addfile(ti, io.BytesIO(data))
    (root / ".nv-meta" / "split.yaml").write_text(
        "split_parts:\n  train:\n  - train_000.tar\n  val:\n  - val_000.tar\nexclude: []\n"
    )
    (root / ".nv-meta" / "dataset.yaml").write_text(
        "sample_type: {__module__: megatron.energon, __class__: NpyTxtJsonSample}\n"
        "field_map: {images: npy, metadata: json}\n"
    )


@pytest.fixture(scope="module")
def eval_set(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("falcon_eval")
    _dataset(tmp / "src")
    blend = tmp / "blend.yaml"
    blend.write_text(
        "splits:\n  train:\n    blend_epochized:\n"
        f"    - repetitions: 1\n      path: {tmp / 'src'}\n"
        f"    - repetitions: 1\n      path: {tmp / 'no_val_split'}\n"
    )
    out = tmp / "eval"
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(SCRIPTS, "make_falcon_eval_set.py"),
            str(blend),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "skipped (no val split" in r.stdout
    return out


def test_converter_output(eval_set):
    gt = json.load(open(eval_set / "gt.json"))
    files = {p["file_name"] for p in gt["images"]}
    assert len(files) == 4 and all("__val" in f for f in files)  # val tars only
    assert all((eval_set / "images" / f).exists() for f in files)
    # Per image: dog (2 masks), cat (1), 2 negatives; the unicorn ([]) and DET are not pairs.
    by_text = {}
    for p in gt["images"]:
        by_text.setdefault(p["text_input"], set()).add(p["num_instances"])
    assert by_text["dog"] == {2} and by_text["sleeping cat on the left"] == {1}
    assert by_text["bicycle"] == {0} and by_text["boat"] == {0}
    assert "unicorn" not in by_text
    assert len(gt["annotations"]) == 4 * 3
    a = gt["annotations"][0]
    assert a["segmentation"]["size"] == [300, 400] and a["bbox"][2] > 0
    counts = {}
    for ann in gt["annotations"]:
        counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1
    assert all(counts.get(p["id"], 0) == p["num_instances"] for p in gt["images"])
    assert {p["level"] for p in gt["images"]} == {"0", "3", "neg:none"}


def _perfect_predictions(gt, path):
    preds = [
        {
            "image_id": a["image_id"],
            "category_id": 1,
            "segmentation": a["segmentation"],
            "bbox": a["bbox"],
            "score": 1.0,
        }
        for a in gt["annotations"]
    ]
    with open(path, "w") as f:
        json.dump(preds, f)


def test_cgf1_by_slice_perfect_predictions(eval_set, tmp_path):
    gt = json.load(open(eval_set / "gt.json"))
    pred = tmp_path / "coco_predictions_segm.json"
    _perfect_predictions(gt, pred)
    out = tmp_path / "slices.json"
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(SCRIPTS, "cgf1_by_slice.py"),
            "--gt",
            str(eval_set / "gt.json"),
            "--pred",
            str(pred),
            "--min-pairs",
            "1",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.load(open(out))
    # cgF1 = F1 x IL_MCC, both smoothed: ~1 - 1e-4 for perfect predictions.
    assert res["all"]["cgF1"] == pytest.approx(1.0, abs=1e-3)
    assert res["level=0"]["cgF1"] == pytest.approx(1.0, abs=1e-3)
    assert "lang=en" in res and "source=" + gt["images"][0]["source"] in res


def test_eval_config_loads_the_heldout_set(eval_set, tmp_path, monkeypatch):
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    try:
        register_omegaconf_resolvers()
    except ValueError:
        pass
    monkeypatch.setenv("FALCON_EVAL_DIR", str(eval_set))
    monkeypatch.setenv("SAM3_EVAL_CHECKPOINT", str(tmp_path / "ckpt.pt"))
    monkeypatch.setenv("SAM3_EVAL_OUT", str(tmp_path / "out"))
    GlobalHydra.instance().clear()
    initialize_config_module("sam3.train", version_base="1.2")
    cfg = compose(
        config_name="configs/falcon_v15/eval/falcon_heldout.yaml",
        overrides=["scratch.num_val_workers=0"],
    )
    assert cfg.trainer.mode == "val"
    assert cfg.trainer.model._target_.endswith("build_sam3_image_model_for_eval")
    ds = instantiate(cfg.trainer.data.val.dataset, _convert_="all")
    gt = json.load(open(eval_set / "gt.json"))
    assert len(ds) == len(gt["images"])
    collate = instantiate(cfg.trainer.data.val.collate_fn)
    batch = collate([ds[0], ds[1]])["falcon_heldout"]
    assert batch.img_batch.shape[-2:] == (1008, 1008)
    assert len(batch.find_text_batch) >= 1
    GlobalHydra.instance().clear()


def test_eval_builder_loads_a_training_checkpoint(tmp_path):
    from sam3.train import model_init

    kw = dict(vit_depth=2, vit_global_att_blocks=(1,), text_context_length=64)
    model = model_init.build_sam3_image_model_for_training(pe_checkpoint=None, **kw)
    sd = model.state_dict()
    torch.save({"model": sd, "step": 7}, tmp_path / "checkpoint.pt")
    del model
    ev = model_init.build_sam3_image_model_for_eval(
        checkpoint_path=str(tmp_path / "checkpoint.pt"),
        device="cpu",
        bpe_path=None,
        enable_segmentation=True,
        eval_mode=True,
        **kw,
    )
    assert not ev.training
    k = "backbone.language_backbone.encoder.positional_embedding"
    torch.testing.assert_close(ev.state_dict()[k], sd[k])
    with pytest.raises(ValueError, match="does not match"):
        model_init.build_sam3_image_model_for_eval(
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            device="cpu",
            vit_depth=4,
            vit_global_att_blocks=(3,),
            text_context_length=64,
        )


def test_cgf1_slices_discriminate(eval_set, tmp_path):
    """Predictions that miss every "cat" mask (level 3) hurt level=3 only."""
    gt = json.load(open(eval_set / "gt.json"))
    cat_ids = {p["id"] for p in gt["images"] if p["level"] == "3"}
    pruned = dict(
        gt, annotations=[a for a in gt["annotations"] if a["image_id"] not in cat_ids]
    )
    pred = tmp_path / "coco_predictions_segm.json"
    _perfect_predictions(pruned, pred)
    out = tmp_path / "slices.json"
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(SCRIPTS, "cgf1_by_slice.py"),
            "--gt",
            str(eval_set / "gt.json"),
            "--pred",
            str(pred),
            "--min-pairs",
            "1",
            "--by",
            "level",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.load(open(out))
    assert res["level=0"]["cgF1"] == pytest.approx(1.0, abs=1e-3)
    assert res["level=3"]["cgF1"] < 0.5
    assert res["all"]["cgF1"] < res["level=0"]["cgF1"]
