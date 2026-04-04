# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import functools
import getpass
import io
import os
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from shlex import quote
from types import SimpleNamespace
from typing import NamedTuple

import datasets
import einops
import numpy as np
import PIL.Image
import torch
from timm.optim.adan import Adan
from torchmetrics.classification import JaccardIndex
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import pil_to_tensor
from tqdm.auto import tqdm

from dedelayed.registry import MODELS

Config = SimpleNamespace

DEFAULT_EVAL_COMPRESSION = {"format": "WEBP", "quality": 85, "lossless": False}
DEFAULT_EVAL_PAST_TICKS = 5
X_REMOTE_LEN = 4


def compute_size(h: int, aspect: float, div: int) -> tuple[int, int]:
    h = int(h)
    w = int(aspect * h)
    return (h // div * div), (w // div * div)


def normalize_uint8(x: torch.Tensor) -> torch.Tensor:
    return x / 255.0


def resize_logits(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return Resize(size, interpolation=PIL.Image.Resampling.BILINEAR)(logits)


def raised_cosine_scheduler(i_step: int, config: Config) -> float:
    t = i_step / config.total_steps
    return (config.max_lr - config.min_lr) * (
        1 - ((np.cos(np.pi * t)) ** (2 * config.lr_pow))
    ) + config.min_lr


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
                p=0.8,
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
    past_ticks = ts.past_ticks

    x_remote_batch = []
    x_local_batch = []
    target_batch = []

    for sample in batch:
        x_remote_i, x_local_i, target_i = augment(
            x_remote_src=[
                sample[f"{config.compression_level}_{i}"] for i in ts.x_remote
            ],
            x_local_src=[sample[f"near_lossless_{ts.x_local}"]],
            target_src=[sample[f"label_{ts.target}"]],
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
    # past_ticks: int frame offset
    return x_remote, x_local, target, past_ticks


def compress_decompress(
    frame: PIL.Image.Image, compression: dict | None
) -> PIL.Image.Image:
    if compression is None:
        return frame.copy()

    with io.BytesIO() as buf:
        frame.save(
            buf,
            format=compression["format"],
            quality=compression["quality"],
            lossless=compression["lossless"],
        )
        buf.seek(0)
        with PIL.Image.open(buf) as img:
            img.load()
            return img.copy()


def preprocess_eval(
    *,
    config: Config,
    x_remote_src: list,
    x_local_src: list,
    compression: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_remote_size = compute_size(config.remote_size, config.aspect, config.ips)
    x_local_size = compute_size(config.local_size, config.aspect, config.ips)

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
    config: Config,
    past_ticks_true: int,
    compression: dict | None,
):
    assert len(batch) == 1
    sample = batch[0]
    assert isinstance(sample, dict)
    idx_eval_frame = config.idx_eval_frame
    x_remote, x_local = preprocess_eval(
        config=config,
        x_remote_src=[
            sample[f"original_{idx_eval_frame - past_ticks_true - k}"]
            for k in reversed(range(X_REMOTE_LEN))
        ],
        x_local_src=[sample[f"original_{idx_eval_frame}"]],
        compression=compression,
    )
    gt = pil_to_tensor(sample[f"label_hq_{idx_eval_frame}"]).squeeze(0)
    x_remote = x_remote.unsqueeze(0)
    x_local = x_local.unsqueeze(0)
    gt = gt.unsqueeze(0)
    return x_remote, x_local, gt


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    device: str,
    dataset: datasets.Dataset,
    *,
    config: Config,
    past_ticks: int,
    past_ticks_residual: int = 0,
    compression: dict | None,
) -> float:
    model.eval()
    past_ticks_true = past_ticks + past_ticks_residual
    assert past_ticks_true >= 0
    metric = JaccardIndex(
        task="multiclass", num_classes=19, average="macro", ignore_index=255
    ).to(device)
    loader = torch.utils.data.DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        collate_fn=functools.partial(
            collate_eval,
            config=config,
            past_ticks_true=past_ticks_true,
            compression=compression,
        ),
    )
    for x_remote, x_local, gt in tqdm(loader, desc="eval", leave=False):
        x_remote = x_remote.to(device)
        x_local = x_local.to(device)
        gt = gt.to(device)
        out = model(x_local[:, :, 0], x_remote, past_ticks)
        gt_h, gt_w = gt.shape[-2:]
        logits = resize_logits(out["seg_logits"], (gt_h, gt_w))
        pred = logits.argmax(dim=1).to(torch.uint8)
        metric.update(pred, gt)
    miou = metric.compute().item()
    return miou


def commit_version(rev: str = "", root: str = ".") -> str:
    cmd_flags = "--long --always --tags --match='v[0-9]*'"
    cmd = f"git -C {quote(root)} describe {cmd_flags}"
    cmd += " --dirty" if rev == "" else f" {quote(rev)}"
    return os.popen(cmd).read().rstrip()


def save_checkpoint(*, meta: dict, runtime: TrainRuntime, state: TrainState) -> None:
    save_dir = Path(meta["checkpoint"]["dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / meta["checkpoint"]["name"]
    ckpt = {
        "meta": meta,
        "logs": {
            "schema": "dedelayed.checkpoint_logs.v1",
            "step": {
                "train/loss": torch.tensor(state.train_losses, dtype=torch.float32),
                "train/lr": torch.tensor(state.learning_rates, dtype=torch.float32),
                "train/grad_norm": torch.tensor(state.grad_norms, dtype=torch.float32),
            },
            "epoch": {
                "val/miou": torch.tensor(state.valid_mious, dtype=torch.float32),
            },
        },
        "state_dict": runtime.model.state_dict(),
    }
    torch.save(ckpt, str(save_path))


@dataclass
class TrainRuntime:
    model: torch.nn.Module
    frozen_modules: list[torch.nn.Module]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    device: str
    config: Config
    dataset: datasets.DatasetDict


@dataclass
class TrainState:
    learning_rates: list[float] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    valid_mious: list[float] = field(default_factory=list)
    last_epoch_idx: int = -1
    global_step: int = 0


def run_epoch(runtime: TrainRuntime, state: TrainState, meta: dict) -> None:
    runtime.model.train()
    for frozen_module in runtime.frozen_modules:
        frozen_module.eval()

    dataloader_train = torch.utils.data.DataLoader(
        runtime.dataset["train"],  # type: ignore[arg-type]
        batch_size=runtime.config.batch_size,
        num_workers=runtime.config.num_workers,
        drop_last=True,
        shuffle=True,
        collate_fn=functools.partial(collate_train, config=runtime.config),
    )

    train_bar = tqdm(
        dataloader_train,
        desc=f"train {state.last_epoch_idx + 1}/{runtime.config.epochs}",
        leave=False,
    )

    for x_remote, x_local, seg_label, past_ticks in train_bar:
        x_remote = x_remote.to(runtime.device)
        x_local = x_local.to(runtime.device)
        seg_label = seg_label.to(runtime.device).to(torch.long)

        out = runtime.model(x_local, x_remote, past_ticks)
        logits = resize_logits(out["seg_logits"], seg_label.shape[-2:])
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

        train_bar.set_postfix(
            loss=f"{state.train_losses[-1]:.3g}",
            lr=f"{state.learning_rates[-1]:.2g}",
        )

    miou = evaluate(
        model=runtime.model,
        device=runtime.device,
        config=runtime.config,
        dataset=runtime.dataset["validation"],
        past_ticks=DEFAULT_EVAL_PAST_TICKS,
        compression=DEFAULT_EVAL_COMPRESSION,
    )
    print(f"miou: {miou}")
    state.valid_mious.append(miou)

    meta["metrics"]["run"]["val_miou"] = miou
    meta["run"]["progress"]["epochs_completed"] = state.last_epoch_idx + 1
    meta["run"]["progress"]["steps_completed"] = state.global_step

    save_checkpoint(meta=meta, runtime=runtime, state=state)


def build_fused_model(model_cfg: dict) -> torch.nn.Module:
    name = model_cfg["name"]
    kw = model_cfg.get("kwargs", {})
    return MODELS[name](
        remote_model=MODELS[f"{name}_remote"](**kw.get("remote_model", {})),
        local_model=MODELS[f"{name}_local"](**kw.get("local_model", {})),
    )


def main() -> None:
    device = "cuda"

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    meta = {
        "schema": "dedelayed.checkpoint_meta.v1",
        "checkpoint": {
            "dir": "checkpoints/",
            "name": (
                "{run[run_id]}."
                "{hp[model][name]}"
                ".r{hp[config][remote_size]}_l{hp[config][local_size]}_ft_bdd100k_e{hp[config][epochs]}"
                ".pth"
            ),
            "parents": [],
        },
        "run": {
            "description": "",
            "argv": sys.argv,
            "criterion": {"key": "metrics.run.val_miou", "mode": "max"},
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
            "progress": {
                "epochs_completed": 0,
                "steps_completed": 0,
            },
            "utc_start_time": datetime.now(timezone.utc).isoformat(),
            "utc_end_time": None,
        },
        "hp": {
            "config": {
                "epochs": 10,
                "batch_size": 1,
                "num_classes": 19,
                "aspect": 1.74,
                "local_size": 480,
                "remote_size": 720,
                "compression_level": "near_lossless",
                "min_delay": 0,
                "max_delay": 5,
                "ips": 32,
                "ops": 8,
                "drop_path": 0.1,
                "max_lr": 1e-4,
                "min_lr": 1e-8,
                "lr_pow": 2,
                "num_workers": len(os.sched_getaffinity(0)),
                "idx_eval_frame": 14,
            },
            "dataset": {
                "train": {
                    "path": "/path/to/labeled_video_dataset",
                    "split": "train",
                },
                "validation": {
                    "path": "/path/to/labeled_video_dataset",
                    "split": "validation",
                },
            },
            "model": {
                "name": "dedelayed_v1_efficientvitl1_efficientvitb0",
                "kwargs": {
                    "remote_model": {
                        "temporal_depth": 4,
                        "temporal_width": 96,
                        "temporal_expand_ratio": 1,
                        "temporal_norm_groups": 32,
                    },
                    "local_model": {},
                },
            },
        },
        "metrics": {
            "run": {
                "val_miou": None,
                "val_miou_at_past_ticks": None,
            }
        },
    }

    meta["checkpoint"]["name"] = meta["checkpoint"]["name"].format_map(meta)
    print(f"Checkpoint name: {meta['checkpoint']['name']}")

    dataset = datasets.DatasetDict(
        {
            key: datasets.load_dataset(value["path"], split=value["split"])
            for key, value in meta["hp"]["dataset"].items()
        }
    )

    config = SimpleNamespace(**meta["hp"]["config"])
    config.total_steps = config.epochs * (
        dataset["train"].num_rows // config.batch_size
    )
    meta["hp"]["config"] = vars(config)

    model = build_fused_model(meta["hp"]["model"])

    model.to(device)
    model.compile(mode="max-autotune")

    for module in model.modules():
        if hasattr(module, "drop_path"):
            module.drop_path = config.drop_path

    frozen_modules: list[torch.nn.Module] = [
        model.remote_model.main_model.image_model,
        model.local_model.image_model,
    ]
    for frozen_module in frozen_modules:
        for param in frozen_module.parameters():
            param.requires_grad_(False)
    for param_name, param in model.named_parameters():
        icon = "🔥 " if param.requires_grad else "❄️ "
        print(f"{icon} {str(list(param.shape)):<24} {param_name}")

    learnable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = Adan(learnable_params, lr=1.0, caution=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda i_step: raised_cosine_scheduler(i_step, config)
    )
    runtime = TrainRuntime(
        model=model,
        frozen_modules=frozen_modules,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        dataset=dataset,
    )
    state = TrainState(
        learning_rates=[optimizer.param_groups[0]["lr"]],
    )

    epoch_bar = tqdm(range(runtime.config.epochs), desc="epoch", leave=True)
    for epoch_idx in epoch_bar:
        state.last_epoch_idx = epoch_idx
        run_epoch(runtime=runtime, state=state, meta=meta)
        epoch_bar.set_postfix(
            loss=f"{state.train_losses[-1]:.3g}",
            miou=f"{state.valid_mious[-1]:.3g}",
            lr=f"{state.learning_rates[-1]:.2g}",
        )

    val_miou_at_past_ticks = []
    for past_ticks in range(6):
        miou = evaluate(
            model=runtime.model,
            device=runtime.device,
            config=runtime.config,
            dataset=runtime.dataset["validation"],
            past_ticks=past_ticks,
            compression=DEFAULT_EVAL_COMPRESSION,
        )
        val_miou_at_past_ticks.append(miou)
    print(f"val_miou_at_past_ticks: {val_miou_at_past_ticks}")

    meta["metrics"]["run"]["val_miou_at_past_ticks"] = val_miou_at_past_ticks
    meta["run"]["utc_end_time"] = datetime.now(timezone.utc).isoformat()

    save_checkpoint(meta=meta, runtime=runtime, state=state)


if __name__ == "__main__":
    main()
