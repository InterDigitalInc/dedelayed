# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from dedelayed.registry import MODELS


def build_fused_model(model_cfg: DictConfig | dict) -> torch.nn.Module:
    model_dict = (
        cast(dict, OmegaConf.to_container(model_cfg, resolve=True))
        if isinstance(model_cfg, DictConfig)
        else dict(model_cfg)
    )
    name = model_dict["name"]
    kw = dict(model_dict.get("kwargs", {}))
    kw["remote_model"] = MODELS[f"{name}_remote"](**kw.get("remote_model", {}))
    kw["local_model"] = MODELS[f"{name}_local"](**kw.get("local_model", {}))
    return MODELS[name](**kw)
