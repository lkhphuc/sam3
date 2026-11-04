# pyre-unsafe
"""Step-based trainer for large-scale SAM3 training on a streaming (energon) loader.

Differences from `sam3.train.trainer.Trainer` (epoch-based, built for small
COCO-JSON finetunes):
  * steps, not epochs: schedulers run on where = step / max_steps; checkpoints
    every `checkpoint.save_every_steps` and on SIGTERM / SIGUSR1 (preemption);
  * the checkpoint holds model, optimizer, step, RNG states and every rank's
    dataloader position (one file per rank for the loader and RNG);
  * no per-step host syncs: loss components accumulate on the GPU and are
    reduced and logged every `logging.log_freq` steps, where the finiteness
    check also runs (a non-finite loss stops training before the next save);
  * bf16 autocast without GradScaler (fp16 keeps the scaler);
  * the next batch is copied to the GPU on a side stream while the current
    step runs (pinned memory, see Sam3FalconBatch.pin_memory);
  * optionally (masks_for_matched_queries_only) instance masks are computed
    only for the matched queries, in the loss, not for all queries;
  * parameters can be frozen by pattern (e.g. tensors that never get a
    gradient), so DDP can run with find_unused_parameters=False.
"""

import contextlib
import fnmatch
import json
import logging
import math
import os
import random
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from sam3.train.optim.optimizer import construct_optimizer
from sam3.train.utils.logger import Logger, setup_logging
from sam3.train.utils.train_utils import get_amp_type, makedir

CORE_LOSS_KEY = "core_loss"


@dataclass
class StreamingCheckpointConf:
    save_dir: str
    save_every_steps: int = 2000
    keep_every_steps: int = 0  # also keep checkpoint_<step>.pt every N steps (0: never)
    resume_from: Optional[str] = (
        None  # model-only init from this file if no checkpoint in save_dir
    )


@dataclass
class StreamingLoggingConf:
    log_dir: str
    log_freq: int = 10
    tensorboard_writer: Any = None
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"


@dataclass
class StreamingDistributedConf:
    backend: str = "nccl"
    timeout_mins: int = 30
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    comms_dtype: Optional[str] = None  # "bfloat16" compresses gradient all-reduce


@dataclass
class ProfilerConf:
    enabled: bool = False
    start_step: int = 20
    num_steps: int = 5
    out_dir: Optional[str] = None


def _unwrap(model):
    return (
        model.module
        if isinstance(model, nn.parallel.DistributedDataParallel)
        else model
    )


class _Prefetcher:
    """Copies batch i+1 to the GPU on a side stream while step i runs."""

    def __init__(self, loader_iter, device, convert):
        self.it = loader_iter
        self.device = device
        self.convert = convert
        self.stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self.next = None
        self.wait_s = 0.0
        self._preload()

    def _preload(self):
        t0 = time.perf_counter()
        raw = next(self.it)
        self.wait_s = time.perf_counter() - t0
        ctx = (
            torch.cuda.stream(self.stream)
            if self.stream is not None
            else contextlib.nullcontext()
        )
        with ctx:
            self.next = (raw, self.convert(raw))

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        raw, batch = self.next
        if self.stream is not None:
            # Tensors made on the side stream are now used on the main stream.
            _record_stream(batch, torch.cuda.current_stream(self.device))
        wait_s = self.wait_s
        self._preload()
        return raw, batch, wait_s


def _record_stream(obj, stream):
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            obj.record_stream(stream)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _record_stream(o, stream)
    elif isinstance(obj, dict):
        for o in obj.values():
            _record_stream(o, stream)
    elif hasattr(obj, "__dataclass_fields__"):
        for name in obj.__dataclass_fields__:
            _record_stream(getattr(obj, name), stream)


class StreamingTrainer:
    def __init__(
        self,
        *,
        model: Dict[str, Any],
        loss: Dict[str, Any],
        data: Dict[str, Any],
        optim: Dict[str, Any],
        checkpoint: Dict[str, Any],
        logging: Dict[str, Any],
        max_steps: int,
        seed: int = 123,
        distributed: Optional[Dict[str, Any]] = None,
        amp_dtype: str = "bfloat16",
        frozen_param_patterns: Optional[List[str]] = None,
        batch_matching_over_layers: bool = True,
        masks_for_matched_queries_only: bool = False,
        act_ckpt_off: Optional[List[str]] = None,
        norm_mean: List[float] = (0.5, 0.5, 0.5),
        norm_std: List[float] = (0.5, 0.5, 0.5),
        profiler: Optional[Dict[str, Any]] = None,
        cuda_allow_tf32: bool = True,
    ):
        self.max_steps = int(max_steps)
        self.ckpt_conf = StreamingCheckpointConf(**checkpoint)
        self.log_conf = StreamingLoggingConf(**logging)
        self.dist_conf = StreamingDistributedConf(**(distributed or {}))
        self.prof_conf = ProfilerConf(**(profiler or {}))
        self.amp_dtype = get_amp_type(amp_dtype) if amp_dtype else None
        self.norm_mean, self.norm_std = list(norm_mean), list(norm_std)

        self._setup_distributed(cuda_allow_tf32)
        makedir(self.log_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.log_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.log_conf.log_level_primary,
            log_level_secondary=self.log_conf.log_level_secondary,
        )
        self._seed_everything(seed)

        self.model = instantiate(model, _convert_="all")
        self._freeze(frozen_param_patterns or [])
        if batch_matching_over_layers:
            _unwrap(self.model).batch_matching_over_layers = True
        if act_ckpt_off:
            from sam3.train.model_init import set_activation_checkpointing

            set_activation_checkpointing(self.model, list(act_ckpt_off))
            logging.info(f"Activation checkpointing off for {list(act_ckpt_off)}")
        if masks_for_matched_queries_only:
            # The seg head returns DeferredMasks; Masks computes only the
            # matched (o2o and o2m) queries' mask logits.
            _unwrap(self.model).masks_for_matched_queries_only = True
        self.model.to(self.device)
        self.loss = instantiate(loss, _convert_="all").to(self.device)
        self.optim = construct_optimizer(
            self.model,
            optim["optimizer"],
            optim.get("options"),
            optim.get("param_group_modifiers"),
        )
        self.gradient_clipper = (
            instantiate(optim["gradient_clip"]) if optim.get("gradient_clip") else None
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.amp_dtype == torch.float16,
        )
        self.tb = (
            Logger(
                _AttrDict(
                    tensorboard_writer=dict(self.log_conf.tensorboard_writer or {})
                    or None
                )
            )
            if self.rank == 0
            else None
        )

        self.step = 0
        self.loader = instantiate(data["train"], _convert_="all")
        self._load_checkpoint()
        self._wrap_ddp()

        self._stop_requested = False

    # -- setup -------------------------------------------------------------

    def _setup_distributed(self, allow_tf32: bool):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # sam3's loggers read LOCAL_RANK / RANK, also when the process group
        # was initialised by the caller without torchrun's environment.
        os.environ.setdefault("LOCAL_RANK", str(self.local_rank))
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            backend = self.dist_conf.backend
        else:
            self.device = torch.device("cpu")
            backend = "gloo"
        if not dist.is_initialized():
            from datetime import timedelta

            os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
            if "RANK" not in os.environ:  # single process, e.g. a CPU smoke test
                os.environ.update(
                    RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1"
                )
                os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
            dist.init_process_group(
                backend, timeout=timedelta(minutes=self.dist_conf.timeout_mins)
            )
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        os.environ.setdefault("RANK", str(self.rank))

    def _seed_everything(self, seed: int):
        s = seed + self.rank
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    def _freeze(self, patterns: List[str]):
        if not patterns:
            return
        frozen = []
        for name, p in self.model.named_parameters():
            if any(fnmatch.fnmatch(name, pat) for pat in patterns):
                p.requires_grad_(False)
                frozen.append(name)
        logging.info(f"Froze {len(frozen)} parameters matching {patterns}")
        if not frozen:
            logging.warning(f"frozen_param_patterns {patterns} matched no parameter")

    def _wrap_ddp(self):
        if self.world_size == 1 and self.device.type != "cuda":
            return
        c = self.dist_conf
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            find_unused_parameters=c.find_unused_parameters,
            gradient_as_bucket_view=c.gradient_as_bucket_view,
            static_graph=c.static_graph,
        )
        if c.comms_dtype is not None:
            from torch.distributed.algorithms import ddp_comm_hooks

            hook = (
                ddp_comm_hooks.default_hooks.bf16_compress_hook
                if get_amp_type(c.comms_dtype) == torch.bfloat16
                else ddp_comm_hooks.default_hooks.fp16_compress_hook
            )
            self.model.register_comm_hook(None, hook)

    def _on_signal(self, signum, _frame):
        logging.warning(
            f"Received signal {signum}: checkpointing at the end of this step"
        )
        self._stop_requested = True

    # -- checkpointing -----------------------------------------------------

    def _ckpt_path(self, name: str) -> str:
        return os.path.join(self.ckpt_conf.save_dir, name)

    def save_checkpoint(self):
        save_dir = self.ckpt_conf.save_dir
        if self.rank == 0:
            makedir(save_dir)
        dist.barrier()
        step = self.step
        # Per rank: dataloader position and RNG states (they differ per rank).
        rank_state = {
            "step": step,
            "dataloader": self.loader.state_dict(),
            "rng": _rng_state(),
        }
        _atomic_save(rank_state, self._ckpt_path(f"rank_state_{self.rank:05d}.pt"))
        if self.rank == 0:
            state = {
                "model": _unwrap(self.model).state_dict(),
                "optimizer": self.optim.optimizer.state_dict(),
                "loss": self.loss.state_dict(),
                "step": step,
                "world_size": self.world_size,
            }
            if self.scaler.is_enabled():
                state["scaler"] = self.scaler.state_dict()
            _atomic_save(state, self._ckpt_path("checkpoint.pt"))
            keep = self.ckpt_conf.keep_every_steps
            if keep and step % keep == 0:
                g_pathmgr.copy(
                    self._ckpt_path("checkpoint.pt"),
                    self._ckpt_path(f"checkpoint_{step}.pt"),
                    overwrite=True,
                )
        dist.barrier()
        logging.info(f"Saved checkpoint at step {step}")

    def _load_checkpoint(self):
        path = self._ckpt_path("checkpoint.pt")
        if not g_pathmgr.isfile(path):
            if self.ckpt_conf.resume_from:
                logging.info(
                    f"Initialising model weights from {self.ckpt_conf.resume_from}"
                )
                with g_pathmgr.open(self.ckpt_conf.resume_from, "rb") as f:
                    sd = torch.load(f, map_location="cpu", weights_only=False)
                _unwrap(self.model).load_state_dict(sd.get("model", sd))
            return
        with g_pathmgr.open(path, "rb") as f:
            state = torch.load(f, map_location="cpu", weights_only=False)
        _unwrap(self.model).load_state_dict(state["model"])
        self.optim.optimizer.load_state_dict(state["optimizer"])
        self.loss.load_state_dict(state["loss"])
        if "scaler" in state and self.scaler.is_enabled():
            self.scaler.load_state_dict(state["scaler"])
        self.step = int(state["step"])
        rank_path = self._ckpt_path(f"rank_state_{self.rank:05d}.pt")
        if state.get("world_size") != self.world_size or not g_pathmgr.isfile(
            rank_path
        ):
            logging.warning(
                f"Resuming model/optimizer at step {self.step}, but the dataloader state is "
                f"not restorable (saved world size {state.get('world_size')}, now "
                f"{self.world_size}): the loader starts from a fresh position."
            )
        else:
            with g_pathmgr.open(rank_path, "rb") as f:
                rank_state = torch.load(f, map_location="cpu", weights_only=False)
            assert rank_state["step"] == self.step, (rank_state["step"], self.step)
            self.loader.load_state_dict(rank_state["dataloader"])
            _set_rng_state(rank_state["rng"])
        logging.info(f"Resumed from {path} at step {self.step}")

    # -- training ----------------------------------------------------------

    def _convert(self, raw):
        return raw.to_batched_datapoint(
            self.device, norm_mean=self.norm_mean, norm_std=self.norm_std
        )

    def _forward_loss(self, batch):
        find_stages = self.model(batch)
        model = _unwrap(self.model)
        find_targets = [model.back_convert(x) for x in batch.find_targets]
        losses = self.loss(find_stages, find_targets)
        return losses

    def run(self):
        # Handlers are installed for the duration of run() only (a bound-method
        # handler would otherwise keep the trainer alive after it returns).
        previous = {
            sig: signal.signal(sig, self._on_signal)
            for sig in (signal.SIGTERM, signal.SIGUSR1)
        }
        try:
            self._run()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _run(self):
        self.model.train()
        prefetch = _Prefetcher(iter(self.loader), self.device, self._convert)
        acc: Dict[str, torch.Tensor] = {}
        acc_steps = 0
        counters = {
            "images": 0,
            "queries": 0,
            "instances": 0,
            "data_wait_s": 0.0,
            "max_data_wait_s": 0.0,
        }
        t_log = time.perf_counter()
        prof = self._maybe_profiler()
        logging.info(f"Training from step {self.step} to {self.max_steps}")
        while self.step < self.max_steps:
            raw, batch, wait_s = next(prefetch)
            counters["images"] += int(raw.images.shape[0])
            counters["queries"] += raw.num_queries
            counters["instances"] += int(raw.boxes.shape[0])
            counters["data_wait_s"] += wait_s
            counters["max_data_wait_s"] = max(counters["max_data_wait_s"], wait_s)

            where = self.step / self.max_steps
            self.optim.step_schedulers(where, step=self.step)
            self.optim.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype or torch.float32,
                enabled=self.amp_dtype is not None,
            ):
                losses = self._forward_loss(batch)
            core = losses.pop(CORE_LOSS_KEY)
            self.scaler.scale(core).backward()
            if self.gradient_clipper is not None:
                self.scaler.unscale_(self.optim.optimizer)
                self.gradient_clipper(model=self.model)
            self.scaler.step(self.optim.optimizer)
            self.scaler.update()
            self.step += 1

            with torch.no_grad():
                acc[CORE_LOSS_KEY] = acc.get(CORE_LOSS_KEY, 0) + core.detach().float()
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        acc[k] = acc.get(k, 0) + v.detach().float()
            acc_steps += 1
            if prof is not None:
                prof.step()

            last = self.step == self.max_steps
            if self.step % self.log_conf.log_freq == 0 or last:
                self._log(acc, acc_steps, counters, time.perf_counter() - t_log)
                acc, acc_steps = {}, 0
                counters = {k: 0 for k in counters}
                t_log = time.perf_counter()
            # A preemption signal reaches ranks at different times; agree on it
            # only at log steps (one small all-reduce + sync), not every step.
            stop = (self.step % self.log_conf.log_freq == 0 or last) and self._agree(
                self._stop_requested
            )
            if stop or last or self.step % self.ckpt_conf.save_every_steps == 0:
                if acc_steps:
                    # Never save after an unchecked step.
                    self._log(acc, acc_steps, counters, time.perf_counter() - t_log)
                    acc, acc_steps = {}, 0
                    counters = {k: 0 for k in counters}
                    t_log = time.perf_counter()
                self.save_checkpoint()
            if stop:
                logging.warning("Stopping after preemption checkpoint")
                break
        if prof is not None:
            prof.stop()

    def _agree(self, flag: bool) -> bool:
        """True on every rank if any rank got a stop signal."""
        t = torch.tensor([1.0 if flag else 0.0], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    def _log(self, acc, acc_steps, counters, elapsed):
        keys = sorted(acc)
        vals = torch.stack([acc[k] for k in keys]) / acc_steps
        dist.all_reduce(vals)
        vals = (vals / self.world_size).tolist()
        stats = dict(zip(keys, vals))
        core = stats[CORE_LOSS_KEY]
        if not math.isfinite(core):
            raise FloatingPointError(f"Non-finite loss {core} at step {self.step}")
        c = torch.tensor(
            [
                counters["images"],
                counters["queries"],
                counters["instances"],
                counters["data_wait_s"],
            ],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(c)
        mx = torch.tensor([counters["max_data_wait_s"]], device=self.device)
        dist.all_reduce(mx, op=dist.ReduceOp.MAX)
        n = max(acc_steps, 1)
        perf = {
            "Perf/step_time_s": elapsed / n,
            "Perf/images_per_s_per_gpu": c[0].item() / elapsed / self.world_size,
            "Perf/queries_per_s_per_gpu": c[1].item() / elapsed / self.world_size,
            "Perf/instances_per_step_per_gpu": c[2].item() / n / self.world_size,
            "Perf/data_wait_s_mean": c[3].item() / n / self.world_size,
            "Perf/data_wait_s_max": mx.item(),
        }
        if self.device.type == "cuda":
            perf["Perf/max_mem_gib"] = (
                torch.cuda.max_memory_allocated(self.device) / 2**30
            )
            torch.cuda.reset_peak_memory_stats(self.device)
        lrs = {
            f"Optim/lr_group{i}": g["lr"]
            for i, g in enumerate(self.optim.optimizer.param_groups)
        }
        if self.rank == 0:
            logging.info(
                f"step {self.step}/{self.max_steps} loss {core:.4f} "
                + " ".join(f"{k.split('/')[-1]}={v:.3g}" for k, v in perf.items())
            )
            if self.tb is not None:
                self.tb.log_dict(
                    {f"Losses/{k}": v for k, v in stats.items()}, self.step
                )
                self.tb.log_dict(perf, self.step)
                self.tb.log_dict(lrs, self.step)
            with g_pathmgr.open(
                os.path.join(self.log_conf.log_dir, "train_stats.jsonl"), "a"
            ) as f:
                f.write(json.dumps({"step": self.step, **stats, **perf}) + "\n")

    def _maybe_profiler(self):
        c = self.prof_conf
        if not c.enabled or self.rank != 0:
            return None
        out = c.out_dir or os.path.join(self.log_conf.log_dir, "profile")
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=max(c.start_step - self.step - 1, 0),
                warmup=1,
                active=c.num_steps,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(out),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        prof.start()
        return prof


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def _atomic_save(obj, path: str):
    tmp = f"{path}.tmp"
    with g_pathmgr.open(tmp, "wb") as f:
        torch.save(obj, f)
    if g_pathmgr.exists(path):
        g_pathmgr.rm(path)
    assert g_pathmgr.mv(tmp, path)
