"""CPU smoke test: the real training config end to end (tiny trunk, fake loader)."""

import json
import os
import random

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from falcon_fixtures import raw_sample
from sam3.train.data.falcon_energon import collate_samples, Sam3FalconTaskEncoder


class FakeLoader:
    """In-memory stand-in for Sam3EnergonLoader (same state_dict contract).

    One query per step: on the CPU the per-query fusion encoder materialises
    72x72-token attention matrices, ~4 GB per query at 1008 px.
    """

    def __init__(self):
        random.seed(0)
        enc = Sam3FalconTaskEncoder(
            resolution=1008,
            resize_scales=[1008],
            max_queries_per_sample=1,
            max_queries_per_batch=1,
            log_stats_every=0,
        )
        fn = type(enc).encode_sample.__wrapped__
        s, _ = raw_sample(key="src/0", h=200, w=300)
        splits = list(fn(enc, s))  # one query each: positives and negatives
        self.batches = [collate_samples([x]) for x in splits]
        assert any(b.boxes.shape[0] == 0 for b in self.batches)  # a negative step
        assert any(b.boxes.shape[0] > 0 for b in self.batches)
        self.pos = 0

    def __iter__(self):
        while True:
            yield self.batches[self.pos % len(self.batches)]
            self.pos += 1

    def state_dict(self):
        return {"dp_rank0": {"pos": self.pos}}

    def load_state_dict(self, state):
        self.pos = state["dp_rank0"]["pos"]


def _cfg(tmp_path, max_steps):
    GlobalHydra.instance().clear()
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    try:
        register_omegaconf_resolvers()
    except ValueError:
        pass  # already registered
    initialize_config_module("sam3.train", version_base="1.2")
    cfg = compose(
        config_name="configs/falcon_v15/sam3_falcon_v15_seg.yaml",
        overrides=[
            f"paths.experiment_log_dir={tmp_path}",
            f"scratch.max_steps={max_steps}",
            "scratch.warmup_steps=1",
            "scratch.cooldown_steps=1",
            "trainer.amp_dtype=null",
            "trainer.model.vit_depth=2",
            "trainer.model.vit_global_att_blocks=[1]",
            "trainer.model.pe_checkpoint=null",
            "trainer.checkpoint.save_every_steps=2",
            "trainer.logging.log_freq=1",
            "trainer.optim.optimizer.fused=false",
        ],
    )
    cfg.trainer.data.train = OmegaConf.create({"_target_": f"{__name__}.FakeLoader"})
    return cfg


@pytest.mark.timeout(1800)
def test_train_checkpoint_resume(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(os.path.dirname(__file__))
    trainer = instantiate(_cfg(tmp_path, 2).trainer, _recursive_=False)
    p0 = {k: v.clone() for k, v in trainer.model.named_parameters() if v.requires_grad}
    trainer.run()
    assert trainer.step == 2
    changed = [
        k
        for k, v in trainer.model.named_parameters()
        if v.requires_grad and not torch.equal(v, p0[k])
    ]
    assert changed, "no parameter was updated"
    del p0
    # Every trainable parameter got a gradient (text-only data: no box/point prompts).
    no_grad = [
        k
        for k, v in trainer.model.named_parameters()
        if v.requires_grad and v.grad is None
    ]
    # The config freezes the parameters that get no gradient on this data, so
    # DDP can run with find_unused_parameters=False.
    assert not no_grad, f"trainable parameters without a gradient: {no_grad}"
    ckpt = tmp_path / "checkpoints"
    assert (ckpt / "checkpoint.pt").exists() and (ckpt / "rank_state_00000.pt").exists()
    stats = [json.loads(l) for l in open(tmp_path / "logs" / "train_stats.jsonl")]
    assert [s["step"] for s in stats] == [1, 2]
    for s in stats:
        assert torch.isfinite(torch.tensor(s["core_loss"]))
        assert s["Perf/queries_per_s_per_gpu"] > 0
    loss_keys = set(stats[0])
    for k in ("loss_mask", "loss_dice", "loss_ce", "loss_bbox", "loss_semantic_seg"):
        assert k in loss_keys, (k, sorted(loss_keys))

    # Resume: step, model and loader position come back; training continues.
    # Checksums, not copies: the CPU box cannot hold a second model next to it.
    sd = {k: v.double().sum().item() for k, v in trainer.model.state_dict().items()}
    pos = trainer.loader.pos
    del trainer  # two models do not fit next to each other in a small CPU box
    import gc

    gc.collect()
    trainer2 = instantiate(_cfg(tmp_path, 3).trainer, _recursive_=False)
    assert trainer2.step == 2
    assert trainer2.loader.pos == pos
    for k, v in trainer2.model.state_dict().items():
        assert v.double().sum().item() == sd[k], k
    trainer2.run()
    assert trainer2.step == 3
    GlobalHydra.instance().clear()
