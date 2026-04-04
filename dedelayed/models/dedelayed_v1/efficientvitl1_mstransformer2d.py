# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import einops
import torch
import torch.nn.functional as F
from torch import Tensor

from dedelayed.functional.normalization import ImageNormalizationKind, renormalize
from dedelayed.layers.splitvid_v10 import PostpoolBlock, PrepoolBlock
from dedelayed.models.backbones.evit_vd import EfficientViTSeg3D
from dedelayed.models.backbones.mstransformer2d import MSTransformer2D
from dedelayed.registry import register_model

from .base import Dedelayed_v1_Fused, Dedelayed_v1_Local, Dedelayed_v1_Remote

RemoteOutputKey = Literal["downlink_features", "downlink_seg_logits"]


@register_model("dedelayed_v1_efficientvitl1_mstransformer2d_remote")
class Dedelayed_v1_EfficientViTL1_MSTransformer2D_Remote(Dedelayed_v1_Remote):
    def __init__(
        self,
        *,
        normalization_src: ImageNormalizationKind = "01",
        normalization_dest: ImageNormalizationKind = "minus1_1",
    ):
        super().__init__()
        self.normalization_src: ImageNormalizationKind = normalization_src
        self.normalization_dest: ImageNormalizationKind = normalization_dest
        self.main_model = EfficientViTSeg3D()
        self.mlp_pre_pool = PrepoolBlock()
        self.mlp_post_pool = PostpoolBlock()

    def forward(
        self,
        x_remote: Tensor | None = None,
        *,
        z_remote: Tensor | None = None,
        x_local_size: tuple[int, int],
        past_ticks: Tensor,
        output_keys: Sequence[RemoteOutputKey] = (
            "downlink_features",
            # "downlink_seg_logits",
        ),
    ):
        H_local, W_local = x_local_size
        output_size = (H_local // 8, W_local // 8)

        if z_remote is None:
            assert x_remote is not None
            z_remote = self.contextualize(x_remote)
        z = z_remote
        z = z + self.main_model.embed_delay(z, past_ticks)
        z = self.main_model.vit3d(z)
        z = einops.rearrange(z, "b c f h w -> b (c f) h w", f=4)

        outputs = {}

        if "downlink_seg_logits" in output_keys:
            outputs["downlink_seg_logits"] = self.main_model.head(z)

        if "downlink_features" in output_keys:
            z = self.mlp_pre_pool(z)
            z = F.adaptive_avg_pool2d(z, output_size=output_size)
            z = self.mlp_post_pool(z)
            outputs["downlink_features"] = z

        return outputs

    def contextualize(self, x: Tensor) -> Tensor:
        x = renormalize(
            x, src=self.normalization_src, dest=self.normalization_dest, channel_dim=1
        )
        return self.main_model.forward_images(x)

    def encode_image(self, x: Tensor) -> Tensor:
        z = einops.rearrange(x, "b c h w -> b c 1 h w")
        z = self.contextualize(z)
        z = einops.rearrange(z, "b c 1 h w -> b c h w")
        return z


@register_model("dedelayed_v1_efficientvitl1_mstransformer2d_local")
class Dedelayed_v1_EfficientViTL1_MSTransformer2D_Local(Dedelayed_v1_Local):
    def __init__(
        self,
        cls_classes=1000,
        seg_classes=19,
        *,
        normalization_src: ImageNormalizationKind = "01",
        normalization_dest: ImageNormalizationKind = "minus1_1",
    ):
        super().__init__()
        self.normalization_src: ImageNormalizationKind = normalization_src
        self.normalization_dest: ImageNormalizationKind = normalization_dest
        self.image_model = MSTransformer2D(
            cls_classes=cls_classes, seg_classes=seg_classes
        )

    def downlink_features_shape(
        self, x_local_size: tuple[int, int]
    ) -> tuple[int, int, int]:
        h_local, w_local = x_local_size
        return (96, h_local // 8, w_local // 8)

    def forward(self, x_local: Tensor, *, downlink_features: Tensor | None = None):
        z = x_local
        z = renormalize(
            z, src=self.normalization_src, dest=self.normalization_dest, channel_dim=1
        )
        z = self.image_model.T1(z)
        z = z if downlink_features is None else z + downlink_features
        y = z
        z = self.image_model.T2(z)
        y = y + self.image_model.P2(z)
        z = self.image_model.T3(z)
        y = y + self.image_model.P3(z)
        y = self.image_model.seg_head(y)
        seg_logits = y
        return {
            "seg_logits": seg_logits,
        }


@register_model("dedelayed_v1_efficientvitl1_mstransformer2d")
class Dedelayed_v1_EfficientViTL1_MSTransformer2D(Dedelayed_v1_Fused):
    def __init__(
        self,
        remote_model: Dedelayed_v1_EfficientViTL1_MSTransformer2D_Remote,
        local_model: Dedelayed_v1_EfficientViTL1_MSTransformer2D_Local,
    ):
        super().__init__()
        self.remote_model = remote_model
        self.local_model = local_model
        self.drop_downlink_features_prob = 0.0

    def forward(self, x_local: Tensor, x_remote: Tensor, past_ticks: Tensor):
        drop_downlink_features = self.training and (
            torch.rand((), device=x_local.device).item()
            < self.drop_downlink_features_prob
        )

        if drop_downlink_features:
            downlink_features = torch.zeros(
                (
                    x_local.shape[0],
                    *self.local_model.downlink_features_shape(x_local.shape[-2:]),
                ),
                device=x_local.device,
                dtype=x_local.dtype,
            )
        else:
            out_remote = self.remote_model(
                x_remote,
                past_ticks=past_ticks,
                x_local_size=x_local.shape[-2:],
            )
            downlink_features = out_remote["downlink_features"].clone()

        out_local = self.local_model(x_local, downlink_features=downlink_features)

        seg_logits = out_local["seg_logits"]

        return {
            "seg_logits": seg_logits,
        }
