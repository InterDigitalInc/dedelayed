# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import functools
import getpass
import os
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, cast
from uuid import uuid4

import einops
import hydra
import numpy as np
import PIL.Image
import torch
from omegaconf import DictConfig, OmegaConf
from timm.optim.adan import Adan
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics.classification import JaccardIndex
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import pil_to_tensor
from tqdm.auto import tqdm

from dedelayed.datasets.hf import decode_image
from dedelayed.registry import DATASETS, MODELS
from dedelayed.utils.git import commit_version
from dedelayed.utils.optim import RaisedCosineLR
from dedelayed.utils.preprocessing import (
    compress_decompress,
    compute_size,
    normalize_uint8,
)
from dedelayed.utils.trackers import Tracker, build_tracker
from dedelayed.utils.utils import cache_by_id, get_attr_by_key

Config = DictConfig

DEFAULT_EVAL_COMPRESSION = {"format": "WEBP", "quality": 85, "lossless": False}
DEFAULT_EVAL_PAST_TICKS = 5
X_REMOTE_LEN = 4


def augment(
    *,
    x_remote_src: list,
    x_local_src: list,
    target_src: list,
    crop_scale: tuple[float, float] = (0.65, 1.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x_remote_src[0].size == x_local_src[0].size == target_src[0].size
    h, w = target_src[0].height, target_src[0].width
    fill: dict = {tv_tensors.Image: 0, tv_tensors.Mask: 255}
    transforms = T.Compose(
        [
            T.RandomHorizontalFlip(0.5),
            T.RandomApply(
                [
                    T.RandomAffine(
                        degrees=[-7.0, 7.0],
                        shear=(-3.0, 3.0, -3.0, 3.0),
                        fill=fill,
                        interpolation=PIL.Image.Resampling.BILINEAR,
                    ),
                ],
                p=0.1,
            ),
            T.RandomResizedCrop(
                size=(h, w),
                scale=crop_scale,
                ratio=((w / h) * 0.75, (w / h) / 0.75),
                interpolation=PIL.Image.Resampling.BICUBIC,
            ),
            T.RandomApply(
                [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.3
            ),
            T.ToPureTensor(),
        ]
    )

    x_remote, x_local, target = transforms(
        [tv_tensors.Image(frame) for frame in x_remote_src],
        [tv_tensors.Image(frame) for frame in x_local_src],
        [tv_tensors.Mask(frame) for frame in target_src],
    )
    x_remote = torch.stack(x_remote)
    x_local = torch.stack(x_local)
    target = torch.stack(target).squeeze(1)

    x_remote_flat = x_remote.reshape(-1, *x_remote.shape[-2:])
    x_local_flat = x_local.reshape(-1, *x_local.shape[-2:])
    target_flat = target.reshape(-1, *target.shape[-2:])
    return x_remote_flat, x_local_flat, target_flat


class TemporalSample(NamedTuple):
    past_ticks: int
    x_remote: list[int]
    x_local: int
    target: int


def sample_temporal_indices(config: Config):
    past_ticks = np.random.choice(range(config.min_delay, config.max_delay + 1))
    i_start = np.random.choice(range(0, (16 - X_REMOTE_LEN) - past_ticks))
    i_frames = list(range(i_start, i_start + past_ticks + X_REMOTE_LEN))
    return TemporalSample(
        past_ticks=past_ticks,
        x_remote=i_frames[:X_REMOTE_LEN],
        x_local=i_frames[-1],
        target=i_frames[-1],
    )


def collate_train(batch, *, config: Config):
    x_remote_size = compute_size(config.remote_size, config.aspect, config.ips)
    x_local_size = compute_size(config.local_size, config.aspect, config.ips)

    ts = sample_temporal_indices(config)
    past_ticks = torch.full((len(batch),), float(ts.past_ticks), dtype=torch.float32)

    x_remote_batch = []
    x_local_batch = []
    target_batch = []

    for sample in batch:
        decode = cache_by_id(decode_image)
        x_remote_i, x_local_i, target_i = augment(
            x_remote_src=[decode(sample["remote_frame"][i]) for i in ts.x_remote],
            x_local_src=[decode(sample["local_frame"][ts.x_local])],
            target_src=[decode(sample["seg_mask"][ts.target])],
        )
        x_remote_batch.append(x_remote_i)
        x_local_batch.append(x_local_i)
        target_batch.append(target_i)

    x_remote = torch.stack(x_remote_batch)
    x_remote = einops.rearrange(x_remote, "b (f c) h w -> (b f) c h w", c=3)
    x_remote = Resize(x_remote_size, interpolation=PIL.Image.Resampling.BICUBIC)(
        x_remote
    )
    x_remote = x_remote.view(len(x_remote_batch), X_REMOTE_LEN, 3, *x_remote.shape[-2:])
    x_remote = x_remote.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W]
    x_remote = normalize_uint8(x_remote)
    x_local = torch.stack(x_local_batch)
    x_local = Resize(x_local_size, interpolation=PIL.Image.Resampling.BICUBIC)(x_local)
    x_local = normalize_uint8(x_local)
    target = torch.stack(target_batch)
    target = target.squeeze(1)

    # x_remote: [B, 3, x_remote_len, H_remote, W_remote], float32
    # x_local: [B, 3, H_local, W_local], float32
    # target: [B, H_target, W_target], uint8/int
    # past_ticks: [B], float32 frame offsets
    return x_remote, x_local, target, past_ticks


def preprocess_eval(
    *,
    x_remote_src: list,
    x_local_src: list,
    x_remote_size: tuple[int, int],
    x_local_size: tuple[int, int],
    compression: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_remote_frames = []
    for frame in x_remote_src:
        remote_frame = compress_decompress(frame, compression)
        x_remote_i = pil_to_tensor(remote_frame)
        x_remote_i = Resize(x_remote_size, interpolation=PIL.Image.Resampling.BICUBIC)(
            x_remote_i
        )
        x_remote_i = normalize_uint8(x_remote_i)
        x_remote_frames.append(x_remote_i)
    x_remote = torch.stack(x_remote_frames, dim=1)

    x_local_frames = []
    for frame in x_local_src:
        x_local_i = pil_to_tensor(frame)
        x_local_i = Resize(x_local_size, interpolation=PIL.Image.Resampling.BICUBIC)(
            x_local_i
        )
        x_local_i = normalize_uint8(x_local_i)
        x_local_frames.append(x_local_i)
    x_local = torch.stack(x_local_frames, dim=1)

    return x_remote, x_local


def collate_eval(
    batch,
    *,
    past_ticks_true: int,
    x_remote_size: tuple[int, int],
    x_local_size: tuple[int, int],
    compression: dict | None,
):
    assert len(batch) == 1
    sample = batch[0]
    assert isinstance(sample, dict)
    decode = cache_by_id(decode_image)
    x_remote, x_local = preprocess_eval(
        x_remote_src=[
            decode(sample["remote_frame"][-past_ticks_true - k])
            for k in reversed(range(X_REMOTE_LEN))
        ],
        x_local_src=[decode(sample["local_frame"][0])],
        x_remote_size=x_remote_size,
        x_local_size=x_local_size,
        compression=compression,
    )
    gt = pil_to_tensor(decode(sample["seg_mask"][0])).squeeze(0)
    x_remote = x_remote.unsqueeze(0)
    x_local = x_local.unsqueeze(0)
    gt = gt.unsqueeze(0)
    return x_remote, x_local, gt


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    device: str,
    dataset: Dataset,
    *,
    config: Config,
    past_ticks: int,
    past_ticks_offset: int = 0,
    compression: dict | None,
) -> float:
    model.eval()
    past_ticks_true = past_ticks + past_ticks_offset
    assert past_ticks_true >= 0
    metric = JaccardIndex(
        task="multiclass",
        num_classes=config.num_classes,
        average="macro",
        ignore_index=255,
    ).to(device)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers
        if config.num_workers is not None
        else len(os.sched_getaffinity(0)),
        collate_fn=functools.partial(
            collate_eval,
            past_ticks_true=past_ticks_true,
            x_remote_size=compute_size(config.remote_size, config.aspect, config.ips),
            x_local_size=compute_size(config.local_size, config.aspect, config.ips),
            compression=compression,
        ),
    )
    for x_remote, x_local, gt in tqdm(loader, desc="eval", leave=False):
        x_remote = x_remote.to(device)
        x_local = x_local.to(device)
        gt = gt.to(device)
        B, *_ = x_remote.shape
        past_ticks_t = torch.full(
            (B,), float(past_ticks), device=device, dtype=torch.float32
        )
        out = model(x_local[:, :, 0], x_remote, past_ticks_t)
        gt_h, gt_w = gt.shape[-2:]
        logits_interp = PIL.Image.Resampling[config.seg_logits_interpolation]
        logits = Resize((gt_h, gt_w), interpolation=logits_interp)(out["seg_logits"])
        pred = logits.argmax(dim=1).to(torch.uint8)
        metric.update(pred, gt)
    miou = metric.compute().item()
    return miou


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
            "learning_rates": torch.tensor(state.learning_rates, dtype=torch.float32),
            "train_losses": torch.tensor(state.train_losses, dtype=torch.float32),
            "grad_norms": torch.tensor(state.grad_norms, dtype=torch.float32),
            "valid_mious": torch.tensor(state.valid_mious, dtype=torch.float32),
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

    def to_list(x) -> list:
        return x.tolist() if isinstance(x, torch.Tensor) else list(x)

    state = TrainState(
        epoch=int(train_state.get("epoch", 0)),
        global_step=int(train_state.get("global_step", 0)),
        learning_rates=to_list(train_state.get("learning_rates", [])),
        train_losses=to_list(train_state.get("train_losses", [])),
        grad_norms=to_list(train_state.get("grad_norms", [])),
        valid_mious=to_list(train_state.get("valid_mious", [])),
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
    learning_rates: list[float] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    valid_mious: list[float] = field(default_factory=list)


def run_epoch(runtime: TrainRuntime, state: TrainState, epoch_bar: tqdm) -> None:
    config = runtime.cfg.hp.config
    log_step_interval = 1 if runtime.cfg.debug else 100

    runtime.model.train()
    for frozen_module in runtime.frozen_modules:
        frozen_module.eval()

    train_bar = tqdm(
        runtime.dataloader["train"],
        desc=f"train {state.epoch + 1}/{config.epochs}",
        leave=False,
    )

    for x_remote, x_local, seg_label, past_ticks in train_bar:
        x_remote = x_remote.to(runtime.device)
        x_local = x_local.to(runtime.device)
        seg_label = seg_label.to(runtime.device).to(torch.long)
        past_ticks = past_ticks.to(runtime.device)

        out = runtime.model(x_local, x_remote, past_ticks)
        logits_interp = PIL.Image.Resampling[config.seg_logits_interpolation]
        logits = Resize(seg_label.shape[-2:], interpolation=logits_interp)(
            out["seg_logits"]
        )
        loss_ce = torch.nn.CrossEntropyLoss(ignore_index=255)(logits, seg_label)
        total_loss = loss_ce

        runtime.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            runtime.model.parameters(), 5.0, norm_type=2.0
        )
        runtime.optimizer.step()
        runtime.scheduler.step()

        state.train_losses.append(total_loss.item())
        state.learning_rates.append(runtime.optimizer.param_groups[0]["lr"])
        state.grad_norms.append(float(grad_norm))
        state.global_step += 1
        if state.global_step % log_step_interval == 0:
            runtime.tracker.log_metrics(
                {
                    "train/step/loss": state.train_losses[-1],
                    "train/step/lr": state.learning_rates[-1],
                    "train/step/grad_norm": state.grad_norms[-1],
                },
                step=state.global_step,
            )

        train_bar.set_postfix(
            loss=f"{state.train_losses[-1]:.3g}",
            lr=f"{state.learning_rates[-1]:.2g}",
        )

    miou = evaluate(
        model=runtime.model,
        device=runtime.device,
        config=config,
        dataset=runtime.dataset["validation"],
        past_ticks=DEFAULT_EVAL_PAST_TICKS,
        compression=DEFAULT_EVAL_COMPRESSION,
    )
    state.valid_mious.append(miou)
    runtime.tracker.log_metrics(
        {
            "epoch": state.epoch + 1,
            "val/epoch/miou": state.valid_mious[-1],
        },
        step=state.global_step,
    )
    runtime.cfg.metrics.run.val_miou = miou

    epoch_bar.set_postfix(
        loss=f"{state.train_losses[-1]:.3g}",
        miou=f"{state.valid_mious[-1]:.3g}",
        lr=f"{state.learning_rates[-1]:.2g}",
    )
    state.epoch += 1
    save_checkpoint(runtime=runtime, state=state)


def build_dataset(dataset_cfg: DictConfig) -> Dataset:
    dataset_dict = cast(dict, OmegaConf.to_container(dataset_cfg, resolve=True))
    return DATASETS[dataset_dict["name"]](**dataset_dict["kwargs"])


def build_fused_model(model_cfg: DictConfig) -> torch.nn.Module:
    model_dict = cast(dict, OmegaConf.to_container(model_cfg, resolve=True))
    name = model_dict["name"]
    kw = model_dict.get("kwargs", {})
    return MODELS[name](
        remote_model=MODELS[f"{name}_remote"](**kw.get("remote_model", {})),
        local_model=MODELS[f"{name}_local"](**kw.get("local_model", {})),
    )


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

    model.drop_downlink_features_prob = cfg.hp.config.drop_downlink_features_prob

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

    model.remote_model.compile(mode="max-autotune")
    model.local_model.compile(mode="max-autotune")

    return model, frozen_modules


@hydra.main(version_base=None, config_path="../conf", config_name="config")
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
            collate_fn=functools.partial(collate_train, config=config),
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

    val_miou_at_past_ticks = []
    for past_ticks in range(6):
        miou = evaluate(
            model=runtime.model,
            device=runtime.device,
            config=config,
            dataset=runtime.dataset["validation"],
            past_ticks=past_ticks,
            compression=DEFAULT_EVAL_COMPRESSION,
        )
        val_miou_at_past_ticks.append(miou)

    cfg.metrics.run.val_miou_at_past_ticks = val_miou_at_past_ticks
    cfg.run.utc_end_time = datetime.now(timezone.utc).isoformat()

    tracker.log_metrics(
        {
            "epoch": state.epoch,
            **{
                f"val/final/miou_at_past_ticks/{i}": val
                for i, val in enumerate(val_miou_at_past_ticks)
            },
        },
        step=state.global_step,
    )

    tracker.close()

    save_checkpoint(runtime=runtime, state=state)


if __name__ == "__main__":
    main()
