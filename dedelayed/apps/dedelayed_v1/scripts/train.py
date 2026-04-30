# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import functools
import getpass
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple, cast
from uuid import uuid4

import einops
import hydra
import numpy as np
import PIL.Image
import torch
from omegaconf import DictConfig, OmegaConf
from timm.optim.adan import Adan
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics.classification import JaccardIndex
from torchvision.transforms.v2 import Resize
from tqdm.auto import tqdm

from dedelayed.apps.dedelayed_v1.preprocess import (
    Clip,
    ClipIdx,
    ComposeTemporal,
    RandomSpeedupShift,
    build_eval_transform,
    build_train_transform,
    preprocess_clip,
    resolve_clip_idx,
)
from dedelayed.apps.dedelayed_v1.train_state import (
    TrainRuntime,
    TrainState,
    restore_training_state,
    save_checkpoint,
)
from dedelayed.datasets.factory import build_dataset
from dedelayed.models.dedelayed_v1.base import Dedelayed_v1_Fused
from dedelayed.models.dedelayed_v1.factory import build_fused_model
from dedelayed.utils.git import commit_version
from dedelayed.utils.optim import RaisedCosineLR
from dedelayed.utils.preprocessing import compute_size
from dedelayed.utils.trackers import build_tracker
from dedelayed.utils.utils import get_attr_by_key

Config = DictConfig

DEFAULT_EVAL_COMPRESSION = {"format": "WEBP", "quality": 85, "lossless": False}
DEFAULT_EVAL_PAST_TICKS = 5
X_REMOTE_LEN = 4


class TemporalSample(NamedTuple):
    idx: ClipIdx
    past_ticks: int
    past_ticks_true: int  # For evaluation only. Models RTT delay mismatch / jitter.
    future_ticks_true: int = 0  # For evaluation only. Models local inference delay.


def sample_temporal_indices_train(config: Config) -> TemporalSample:
    past_ticks = np.random.choice(range(config.min_delay, config.max_delay + 1))
    return sample_temporal_indices_eval(
        past_ticks=past_ticks,
        past_ticks_true=past_ticks,
    )


def sample_temporal_indices_eval(
    past_ticks: int,
    past_ticks_true: int,
    future_ticks_true: int = 0,
) -> TemporalSample:
    return TemporalSample(
        idx=ClipIdx(
            x_remote=[-past_ticks_true - k for k in reversed(range(X_REMOTE_LEN))],
            x_local=[0],
            target=[future_ticks_true],
        ),
        past_ticks=past_ticks,
        past_ticks_true=past_ticks_true,
        future_ticks_true=future_ticks_true,
    )


class CollatedBatch(NamedTuple):
    x_remote: Tensor
    x_local: Tensor
    target: Tensor
    past_ticks: Tensor

    def to(self, device: str | torch.device) -> CollatedBatch:
        return CollatedBatch(
            x_remote=self.x_remote.to(device),
            x_local=self.x_local.to(device),
            target=self.target.to(device),
            past_ticks=self.past_ticks.to(device),
        )


def collate(
    batch: list[dict],
    *,
    sample_temporal_indices: Callable[[], TemporalSample],
    temporal_transform: Callable[[ClipIdx], ClipIdx],
    preprocess_clip: Callable[[dict, ClipIdx], Clip],
) -> CollatedBatch:
    ts = sample_temporal_indices()

    x_remote_batch = []
    x_local_batch = []
    target_batch = []
    past_ticks_batch = []

    for sample in batch:
        idx = resolve_clip_idx(
            ts.idx,
            sample,
            past_ticks_true=ts.past_ticks_true,
            future_ticks_true=ts.future_ticks_true,
        )
        idx = temporal_transform(idx)
        clip = preprocess_clip(sample, idx)
        past_ticks_i = torch.tensor(ts.past_ticks, dtype=torch.float32)
        x_remote_batch.extend(clip.x_remote)
        x_local_batch.extend(clip.x_local)
        target_batch.extend(clip.target)
        past_ticks_batch.append(past_ticks_i)

    x_remote = torch.stack(x_remote_batch)
    x_local = torch.stack(x_local_batch)
    target = torch.stack(target_batch)
    past_ticks = torch.stack(past_ticks_batch)

    B = len(batch)
    x_remote = einops.rearrange(x_remote, "(b f) c h w -> b c f h w", b=B).contiguous()
    x_local = einops.rearrange(x_local, "(b f) c h w -> b c f h w", b=B).contiguous()
    target = einops.rearrange(target, "(b f) h w -> b f h w", b=B).contiguous()

    # x_remote: [B, 3, x_remote_len, H_remote, W_remote], float32
    # x_local: [B, 3, x_local_len, H_local, W_local], float32
    # target: [B, target_len, H_target, W_target], uint8/int
    # past_ticks: [B], float32 frame offsets
    return CollatedBatch(x_remote, x_local, target, past_ticks)


@torch.inference_mode()
def evaluate_dedelayed_v1_segmentation(
    model: torch.nn.Module,
    device: str,
    dataset: Dataset,
    *,
    past_ticks: int = 0,
    past_ticks_offset: int = 0,
    future_ticks_true: int = 0,
    local_only: bool = False,
    uplink_compression: dict | None,
    x_remote_size: tuple[int, int],
    x_local_size: tuple[int, int],
    logits_interp: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC,
    num_workers: int = 0,
) -> float:
    model.eval()
    past_ticks_true = past_ticks + past_ticks_offset
    assert past_ticks_true >= 0
    assert future_ticks_true >= 0
    metric = JaccardIndex(
        task="multiclass",
        num_classes=model.num_classes,
        average="macro",
        ignore_index=255,
    ).to(device)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=functools.partial(
            collate,
            sample_temporal_indices=functools.partial(
                sample_temporal_indices_eval,
                past_ticks,
                past_ticks_true,
                future_ticks_true,
            ),
            temporal_transform=ComposeTemporal([]),
            preprocess_clip=functools.partial(
                preprocess_clip,
                uplink_compression=uplink_compression,
                transform=build_eval_transform(),
                x_remote_size=x_remote_size,
                x_local_size=x_local_size,
            ),
        ),
    )
    for batch in tqdm(loader, desc="eval", leave=False):
        assert isinstance(batch, CollatedBatch)
        batch = batch.to(device)
        out = model(
            batch.x_local[:, :, -1],
            batch.x_remote,
            batch.past_ticks,
            local_only=local_only,
        )
        target = batch.target[:, 0]
        gt_h, gt_w = target.shape[-2:]
        logits = Resize((gt_h, gt_w), interpolation=logits_interp)(out["seg_logits"])
        pred = logits.argmax(dim=1).to(torch.uint8)
        metric.update(pred, target)
    miou = metric.compute().item()
    return miou


def run_epoch(runtime: TrainRuntime, state: TrainState, epoch_bar: tqdm) -> None:
    config = runtime.cfg.hp.config
    model = runtime.model
    assert isinstance(model, Dedelayed_v1_Fused)
    logits_interp = PIL.Image.Resampling[config.seg_logits_interpolation]
    log_step_interval = 1 if runtime.cfg.debug else 100
    loss = torch.tensor(float("nan"))
    lr = runtime.optimizer.param_groups[0]["lr"]
    grad_norm = torch.tensor(float("nan"))
    metrics = {}

    runtime.model.train()
    for frozen_module in runtime.frozen_modules:
        frozen_module.eval()

    train_bar = tqdm(
        runtime.dataloader["train"],
        desc=f"train {state.epoch + 1}/{config.epochs}",
        leave=False,
    )
    num_batches = len(runtime.dataloader["train"])

    for i_batch, batch in enumerate(train_bar):
        assert isinstance(batch, CollatedBatch)
        batch = batch.to(runtime.device)
        torch.compiler.cudagraph_mark_step_begin()
        x_local = batch.x_local[:, :, -1]
        x_local_size = x_local.shape[-2:]
        drop_downlink_features = (
            torch.rand((), device=runtime.device).item()
            < config.drop_downlink_features_prob
        )

        if drop_downlink_features:
            downlink_shape = model.local_model.downlink_features_shape(x_local_size)
            downlink_features = torch.zeros(
                (x_local.shape[0], *downlink_shape),
                device=x_local.device,
                dtype=x_local.dtype,
            )
        else:
            out_remote = model.remote_model(
                batch.x_remote,
                past_ticks=batch.past_ticks,
                x_local_size=x_local_size,
            )
            downlink_features = out_remote["downlink_features"].clone()

        out = model.local_model(x_local, downlink_features)
        target = batch.target[:, 0]
        gt_h, gt_w = target.shape[-2:]
        logits = Resize((gt_h, gt_w), interpolation=logits_interp)(out["seg_logits"])
        loss_ce = torch.nn.CrossEntropyLoss(ignore_index=255)(logits, target)
        loss = loss_ce

        runtime.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.0, norm_type=2.0
        )
        lr = runtime.optimizer.param_groups[0]["lr"]
        runtime.optimizer.step()
        runtime.scheduler.step()

        metrics = {
            "train/step/loss": float(loss),
            "train/step/lr": lr,
            "train/step/grad_norm": float(grad_norm),
        }

        state.global_step += 1

        if state.global_step % log_step_interval == 0 or i_batch + 1 == num_batches:
            runtime.tracker.log_metrics(metrics, step=state.global_step)

        train_bar.set_postfix(
            loss=f"{metrics['train/step/loss']:.3g}",
            lr=f"{metrics['train/step/lr']:.2g}",
        )

    eval_specs = [
        *[
            {
                "metric_name": f"val/epoch/miou_at_past_ticks/{past_ticks}",
                "past_ticks": past_ticks,
            }
            for past_ticks in range(config.min_delay, config.max_delay + 1)
        ],
        {
            "metric_name": "val/epoch/miou_local_only",
            "local_only": True,
        },
    ]

    for spec in eval_specs:
        metrics[spec["metric_name"]] = evaluate_dedelayed_v1_segmentation(
            model=runtime.model,
            device=runtime.device,
            dataset=runtime.dataset["validation"],
            past_ticks=spec.get("past_ticks", 0),
            local_only=spec.get("local_only", False),
            uplink_compression=DEFAULT_EVAL_COMPRESSION,
            x_remote_size=compute_size(config.remote_size, config.aspect, config.ips),
            x_local_size=compute_size(config.local_size, config.aspect, config.ips),
            logits_interp=PIL.Image.Resampling[config.seg_logits_interpolation],
            num_workers=(
                config.num_workers
                if config.num_workers is not None
                else len(os.sched_getaffinity(0))
            ),
        )

    metrics["val/epoch/miou"] = metrics[
        f"val/epoch/miou_at_past_ticks/{DEFAULT_EVAL_PAST_TICKS}"
    ]

    runtime.tracker.log_metrics(
        {
            "epoch": state.epoch + 1,
            **{k: v for k, v in metrics.items() if k.startswith("val/epoch/")},
        },
        step=state.global_step,
    )
    runtime.cfg.metrics.run.val_miou = metrics["val/epoch/miou"]
    runtime.cfg.metrics.run.val_miou_at_past_ticks = [
        metrics[f"val/epoch/miou_at_past_ticks/{past_ticks}"]
        for past_ticks in range(config.min_delay, config.max_delay + 1)
    ]
    runtime.cfg.metrics.run.val_miou_local_only = metrics["val/epoch/miou_local_only"]

    epoch_bar.set_postfix(
        loss=f"{metrics['train/step/loss']:.3g}",
        miou=f"{metrics['val/epoch/miou']:.3g}",
        lr=f"{metrics['train/step/lr']:.2g}",
    )
    state.epoch += 1
    save_checkpoint(runtime=runtime, state=state)


def init_model(
    cfg: DictConfig, device: str, resume_ckpt: dict | None
) -> tuple[torch.nn.Module, list[torch.nn.Module]]:
    model = build_fused_model(cfg.hp.model)

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
    else:
        for parent in cfg.checkpoint.parents:
            ckpt_path = Path(cfg.checkpoint.dir) / parent.path
            parent_ckpt = torch.load(ckpt_path, map_location="cpu")
            submodule = get_attr_by_key(model, parent.key)
            submodule.load_state_dict(parent_ckpt["model_state_dict"], strict=True)

    model.to(device)

    for module in model.modules():
        if hasattr(module, "drop_path"):
            module.drop_path = cfg.hp.config.drop_path

    frozen_modules: list[torch.nn.Module] = [
        model.remote_model.main_model.image_model,
        # model.local_model.image_model,
        *[
            submodule
            for module in [
                model.remote_model.main_model.image_model,
                model.local_model.image_model,
            ]
            for submodule in module.modules()
            if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm)
        ],
    ]
    for frozen_module in frozen_modules:
        for param in frozen_module.parameters():
            param.requires_grad_(False)
    for param_name, param in model.named_parameters():
        icon = "🔥 " if param.requires_grad else "❄️ "
        print(f"{icon} {str(list(param.shape)):<24} {param_name}")

    compile_kwargs: dict = {"mode": "max-autotune"}
    model.remote_model.compile(**compile_kwargs)
    model.local_model.compile(**compile_kwargs)
    model.compile(**compile_kwargs)

    return model, frozen_modules


@hydra.main(version_base=None, config_path="../../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = "cuda"

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    cfg_update = {
        "run": {
            "run_id": f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}",
            "argv": sys.argv,
            "git": {"version": commit_version()},
            "system": {
                "hostname": socket.gethostname(),
                "username": getpass.getuser(),
                "cwd": os.getcwd(),
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "",
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
        },
    }
    cfg = cast(DictConfig, OmegaConf.merge(cfg, cfg_update))

    print(f"Checkpoint name: {cfg.checkpoint.name}")
    ckpt_path = Path(cfg.checkpoint.dir) / cfg.checkpoint.name
    resume_ckpt = None
    if ckpt_path.exists():
        resume_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = cast(DictConfig, OmegaConf.create(resume_ckpt["meta"]))
        print(f"Resuming from checkpoint: {ckpt_path}")

    dataset = {key: build_dataset(ds_cfg) for key, ds_cfg in cfg.hp.dataset.items()}

    config = cfg.hp.config

    if cfg.debug:
        config.epochs = 3
        n_debug = 3 * config.batch_size
        dataset["train"] = Subset(dataset["train"], range(n_debug))
        dataset["validation"] = Subset(dataset["validation"], range(n_debug))

    meta = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    tracker_hparams = {
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
    tracker = build_tracker(cfg.tracker, run_id=cfg.run.run_id, hparams=tracker_hparams)

    source_size = compute_size(config.remote_size, config.aspect, 1)
    x_remote_size = compute_size(config.remote_size, config.aspect, config.ips)
    x_local_size = compute_size(config.local_size, config.aspect, config.ips)
    dataloader = {
        "train": DataLoader(
            dataset["train"],
            batch_size=config.batch_size,
            num_workers=(
                config.num_workers
                if config.num_workers is not None
                else len(os.sched_getaffinity(0))
            ),
            drop_last=True,
            shuffle=True,
            collate_fn=functools.partial(
                collate,
                sample_temporal_indices=functools.partial(
                    sample_temporal_indices_train, config
                ),
                temporal_transform=ComposeTemporal(
                    [
                        RandomSpeedupShift(**cfg.hp.config.random_speedup_shift),
                    ]
                ),
                preprocess_clip=functools.partial(
                    preprocess_clip,
                    uplink_compression=None,
                    transform=build_train_transform(source_size=source_size),
                    x_remote_size=x_remote_size,
                    x_local_size=x_local_size,
                ),
            ),
            persistent_workers=True,
        )
    }
    model, frozen_modules = init_model(cfg, device, resume_ckpt)
    learnable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = Adan(learnable_params, lr=cfg.hp.optim.max_lr, caution=True)
    scheduler = RaisedCosineLR(
        optimizer,
        num_training_steps=config.epochs * len(dataloader["train"]),
        lr_pow=cfg.hp.optim.lr_pow,
    )

    runtime = TrainRuntime(
        model=model,
        frozen_modules=frozen_modules,
        optimizer=optimizer,
        scheduler=scheduler,
        tracker=tracker,
        device=device,
        cfg=cfg,
        dataset=dataset,
        dataloader=dataloader,
    )
    state = restore_training_state(
        optimizer=optimizer,
        scheduler=scheduler,
        ckpt=resume_ckpt,
    )

    epoch_bar = tqdm(
        range(state.epoch, config.epochs),
        initial=state.epoch,
        total=config.epochs,
        desc="epoch",
        leave=True,
    )
    for epoch in epoch_bar:
        assert state.epoch == epoch
        run_epoch(runtime=runtime, state=state, epoch_bar=epoch_bar)

    cfg.run.utc_end_time = datetime.now(timezone.utc).isoformat()

    tracker.close()

    save_checkpoint(runtime=runtime, state=state)


if __name__ == "__main__":
    main()
