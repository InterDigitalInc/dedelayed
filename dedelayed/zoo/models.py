# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple
from urllib.request import urlretrieve

import torch

from dedelayed.models.dedelayed_v1.factory import build_fused_model

ROOT_URL = "https://github.com/InterDigitalInc/dedelayed/releases/download/weights"

CHECKPOINTS = {
    "dedelayed_v1_efficientvitl1_mstransformer2d_bdd100k": "dedelayed_v1_efficientvitl1_mstransformer2d.r720_l480_ft_bdd100k_e10.rev1.pth",
}


class LoadedModel(NamedTuple):
    model: torch.nn.Module
    metadata: dict[str, Any]


def get_root() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / "dedelayed"


def _download_if_needed(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        urlretrieve(url, path)


def get_model(
    name: str,
    *,
    pretrained: bool = True,
    root: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedModel:
    root = Path(root).expanduser() if root is not None else get_root()

    checkpoint_name = CHECKPOINTS[name]
    checkpoint_url = f"{ROOT_URL}/{checkpoint_name}"
    checkpoint_path = root / checkpoint_name

    metadata_name = Path(checkpoint_name).with_suffix(".meta.json").name
    metadata_url = f"{ROOT_URL}/{metadata_name}"
    metadata_path = checkpoint_path.with_suffix(".meta.json")

    _download_if_needed(metadata_path, metadata_url)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
        assert isinstance(metadata, dict)

    model_cfg = metadata["hp"]["model"]
    model = build_fused_model(model_cfg)

    if pretrained:
        _download_if_needed(checkpoint_path, checkpoint_url)
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        assert ckpt["meta"]["hp"]["model"] == model_cfg
        model.load_state_dict(ckpt["model_state_dict"])

    return LoadedModel(model=model, metadata=metadata)


def get_model_from_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> LoadedModel:
    checkpoint_path = Path(path).expanduser()
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    metadata = ckpt["meta"]
    model = build_fused_model(metadata["hp"]["model"])
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    return LoadedModel(model=model, metadata=metadata)
