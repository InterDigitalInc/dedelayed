# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import cast

import PIL.Image
import torch
from omegaconf import DictConfig, OmegaConf

from dedelayed.apps.dedelayed_v1.scripts.train import (
    DEFAULT_EVAL_COMPRESSION,
    evaluate_dedelayed_v1_segmentation,
)
from dedelayed.datasets.factory import build_dataset
from dedelayed.models.dedelayed_v1.base import Dedelayed_v1_Fused
from dedelayed.utils.preprocessing import compute_size
from dedelayed.zoo import get_model, get_model_from_checkpoint


def init_model(args):
    if args.checkpoint is not None:
        model, metadata = get_model_from_checkpoint(args.checkpoint, strict=True)
    else:
        model, metadata = get_model(args.zoo_model, pretrained=True)

    model.to(args.device).eval()
    assert isinstance(model, Dedelayed_v1_Fused)

    if args.compile_enabled:
        compile_kwargs: dict = {"mode": "max-autotune"}
        model.remote_model.compile(**compile_kwargs)
        model.local_model.compile(**compile_kwargs)

    return model, metadata


def parse_args(argv=None) -> argparse.Namespace:
    def expand_path(path: str) -> Path:
        return Path(path).expanduser()

    def add_arguments(parser, arguments: list[dict]) -> None:
        for argument in arguments:
            names = argument["name"]
            aliases = [
                name.replace("_", "-")
                for name in names
                if name.startswith("--") and "_" in name
            ]
            kwargs = {k: v for k, v in argument.items() if k != "name"}
            parser.add_argument(*names, *aliases, **kwargs)

    model_arguments = [
        {"name": ["--zoo_model"], "help": "Model name."},
        {"name": ["--checkpoint"], "type": expand_path, "help": "Checkpoint path."},
    ]
    arguments = [
        {"name": ["--dataset"], "required": True},
        {"name": ["--past_ticks"], "type": int, "nargs": "+", "required": True},
        {"name": ["--future_ticks_true"], "type": int, "default": 0},
        {"name": ["--device"], "default": "cuda"},
        {
            "name": ["--compile"],
            "dest": "compile_enabled",
            "action": argparse.BooleanOptionalAction,
            "default": True,
        },
        {"name": ["--num_workers"], "type": int},
    ]
    parser = argparse.ArgumentParser()
    add_arguments(parser.add_mutually_exclusive_group(required=True), model_arguments)
    add_arguments(parser, arguments)
    args = parser.parse_args(argv)
    return args


def main() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    args = parse_args()

    repo = Path(__file__).resolve().parents[4]
    model, metadata = init_model(args)
    cfg_dict = {
        "hp": {
            "config": metadata["hp"]["config"],
            "model": metadata["hp"]["model"],
            "dataset": OmegaConf.load(repo / f"conf/dataset/{args.dataset}.yaml"),
        },
    }
    cfg = cast(DictConfig, OmegaConf.create(cfg_dict))
    config = cfg.hp.config

    logits_interp = PIL.Image.Resampling[config.seg_logits_interpolation]
    x_remote_size = compute_size(config.remote_size, config.aspect, config.ips)
    x_local_size = compute_size(config.local_size, config.aspect, config.ips)

    dataset = build_dataset(cfg.hp.dataset.validation)

    local_only = True
    miou_local_only = float("nan")

    print(
        f"checkpoint: {args.checkpoint}"
        if args.checkpoint is not None
        else f"zoo_model: {args.zoo_model}"
    )
    print(f"dataset: {cfg.hp.dataset.validation.kwargs.path}")
    print("past_ticks\tmiou\tmiou_remote_only\tmiou_local_only")

    for past_ticks in args.past_ticks:
        metrics = evaluate_dedelayed_v1_segmentation(
            model=model,
            device=args.device,
            dataset=dataset,
            past_ticks=past_ticks,
            future_ticks_true=args.future_ticks_true,
            local_only=local_only,
            uplink_compression=DEFAULT_EVAL_COMPRESSION,
            x_remote_size=x_remote_size,
            x_local_size=x_local_size,
            logits_interp=logits_interp,
            num_workers=args.num_workers
            if args.num_workers is not None
            else len(os.sched_getaffinity(0)),
        )
        if local_only:
            miou_local_only = metrics["miou_local_only"]
            local_only = False
        print(
            f"{past_ticks}\t"
            f"{metrics['miou']:.6f}\t"
            f"{metrics['miou_remote_only']:.6f}\t"
            f"{miou_local_only:.6f}"
        )


if __name__ == "__main__":
    main()
