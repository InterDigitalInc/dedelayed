# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from dedelayed.utils.trackers import Tracker


def save_checkpoint(*, runtime: TrainRuntime, state: TrainState) -> None:
    cfg = runtime.cfg
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
