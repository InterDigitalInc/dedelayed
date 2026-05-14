# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import getpass
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from uuid import uuid4

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from dedelayed.datasets.factory import build_dataset
from dedelayed.utils.git import commit_version
from dedelayed.utils.trackers import Tracker


@dataclass
class TrainRuntime:
    model: torch.nn.Module
    frozen_modules: list[torch.nn.Module]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    tracker: Tracker
    device: str
    cfg: DictConfig
    dataset: dict[str, Dataset]
    dataloader: dict[str, DataLoader]


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0


def update_run_start_metadata(cfg: DictConfig) -> None:
    run_update = {
        "run_id": f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}",
        "argv": sys.argv,
        "git": {"version": commit_version()},
        "system": {
            "hostname": socket.gethostname(),
            "username": getpass.getuser(),
            "cwd": os.getcwd(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        },
        "slurm": {
            "job_id": (
                f"{os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID', '')}_"
                f"{os.environ.get('SLURM_ARRAY_TASK_ID', '')}"
            ).rstrip("_"),
            "job_name": os.environ.get("SLURM_JOB_NAME", ""),
        },
        "utc_start_time": datetime.now(timezone.utc).isoformat(),
        "utc_end_time": None,
    }
    OmegaConf.update(cfg, "run", run_update, merge=True)


def update_run_end_metadata(cfg: DictConfig) -> None:
    cfg.run.utc_end_time = datetime.now(timezone.utc).isoformat()


def build_tracker_hparams(cfg: DictConfig) -> dict:
    meta = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    return {
        "run": {
            "run_id": meta["run"]["run_id"],
            "description": meta["run"]["description"],
            "argv": meta["run"]["argv"],
            "git": meta["run"]["git"],
            "slurm": meta["run"]["slurm"],
        },
        "checkpoint": meta["checkpoint"],
        "hp": meta["hp"],
    }


def load_resume_checkpoint(cfg: DictConfig) -> tuple[DictConfig, dict | None]:
    print(f"Checkpoint name: {cfg.checkpoint.name}")
    ckpt_path = Path(cfg.checkpoint.dir) / cfg.checkpoint.name
    if not ckpt_path.exists():
        return cfg, None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = cast(DictConfig, OmegaConf.create(ckpt["meta"]))
    print(f"Resuming from checkpoint: {ckpt_path}")
    return cfg, ckpt


def save_checkpoint(*, runtime: TrainRuntime, state: TrainState) -> None:
    cfg = runtime.cfg
    cfg.metrics.run.epoch = state.epoch
    cfg.metrics.run.global_step = state.global_step
    meta = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    save_dir = Path(cfg.checkpoint.dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / cfg.checkpoint.name
    ckpt = {
        "meta": meta,
        "model_state_dict": runtime.model.state_dict(),
        "optimizer_state_dict": runtime.optimizer.state_dict(),
        "scheduler_state_dict": runtime.scheduler.state_dict(),
        "train_state": {
            "epoch": state.epoch,
            "global_step": state.global_step,
        },
    }
    torch.save(ckpt, str(save_path))


def restore_training_state(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    ckpt: dict | None,
) -> TrainState:
    ckpt = ckpt or {"train_state": {}}
    train_state = ckpt["train_state"]
    state = TrainState(
        epoch=int(train_state.get("epoch", 0)),
        global_step=int(train_state.get("global_step", 0)),
    )
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return state


def build_experiment_datasets(cfg: DictConfig) -> dict[str, Dataset]:
    dataset = {key: build_dataset(ds_cfg) for key, ds_cfg in cfg.hp.dataset.items()}

    if cfg.debug:
        config = cfg.hp.config
        config.epochs = 3
        n_debug = 64 * config.batch_size
        dataset["train"] = Subset(dataset["train"], range(n_debug))
        dataset["validation"] = Subset(dataset["validation"], range(n_debug))

    return dataset
