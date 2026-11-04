# pyre-unsafe
"""Entry point for step-based SAM3 training (StreamingTrainer), launched by torchrun.

    torchrun --nnodes $NNODES --nproc_per_node 8 --rdzv_backend c10d \\
        --rdzv_endpoint $MASTER_ADDR:29500 -m sam3.train.train_streaming \\
        -c configs/falcon_v15/sam3_falcon_v15_seg.yaml \\
        paths.experiment_log_dir=/mnt/.../sam3_runs/run1

Extra arguments are Hydra overrides (key=value).
"""

import os
from argparse import ArgumentParser

from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from sam3.train.utils.train_utils import makedir, register_omegaconf_resolvers


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    register_omegaconf_resolvers()
    initialize_config_module("sam3.train", version_base="1.2")
    cfg = compose(config_name=args.config, overrides=args.overrides)
    log_dir = cfg.paths.experiment_log_dir
    if int(os.environ.get("RANK", "0")) == 0:
        makedir(log_dir)
        with g_pathmgr.open(os.path.join(log_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        with g_pathmgr.open(os.path.join(log_dir, "config_resolved.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


if __name__ == "__main__":
    main()
