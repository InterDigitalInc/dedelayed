# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import einops
import torch
import torch.nn.functional as F
from torch import Tensor

from dedelayed.functional.normalization import ImageNormalizationKind, renormalize
from dedelayed.layers.splitvid_v10 import PostpoolBlock, PrepoolBlock
from dedelayed.models.backbones.evit_vd import EfficientViTSeg3D
from dedelayed.registry import register_model

from .base import Dedelayed_v1_Fused, Dedelayed_v1_Local, Dedelayed_v1_Remote

RemoteOutputKey = Literal["downlink_features", "downlink_seg_logits"]


@register_model("dedelayed_v1_efficientvitl1_efficientvitb0_remote")
class Dedelayed_v1_EfficientViTL1_EfficientViTB0_Remote(Dedelayed_v1_Remote):
    def __init__(
        self,
        name="efficientvit-seg-l1-cityscapes",
        *,
        normalization_src: ImageNormalizationKind = "01",
        normalization_dest: ImageNormalizationKind = "imagenet",
        temporal_depth: int = 4,
        temporal_width: int = 96,
        temporal_expand_ratio: int = 1,
        temporal_norm_groups: int = 32,
        downlink_channels: int = 32,
    ):
        super().__init__()
        self.normalization_src: ImageNormalizationKind = normalization_src
        self.normalization_dest: ImageNormalizationKind = normalization_dest
        self.main_model = EfficientViTSeg3D(
            name=name,
            temporal_depth=temporal_depth,
            temporal_width=temporal_width,
            temporal_expand_ratio=temporal_expand_ratio,
            temporal_norm_groups=temporal_norm_groups,
            pretrained_image_model=True,
        )
        self.mlp_pre_pool = PrepoolBlock(in_channels=temporal_width * 4)
        self.mlp_post_pool = PostpoolBlock(out_channels=downlink_channels)

    def forward(
        self,
        x_remote: Tensor | None = None,
        *,
        z_encoded: Tensor | None = None,
        x_local_size: tuple[int, int],
        past_ticks: Tensor,
        output_keys: Sequence[str] = (
            "downlink_features",
            # "downlink_seg_logits",
        ),
    ) -> dict[str, Tensor]:
        if z_encoded is None:
            assert x_remote is not None
            z_encoded = self.encode_frames(x_remote)
        z = self.blend(z_encoded)
        z = self.prealign(z, past_ticks)
        return self.head(z, x_local_size=x_local_size, output_keys=output_keys)

    def encode_frames(self, x_remote: Tensor) -> Tensor:
        x_remote = renormalize(
            x_remote,
            src=self.normalization_src,
            dest=self.normalization_dest,
            channel_dim=1,
        )
        return self.main_model.forward_images(x_remote)

    def blend(self, z_encoded: Tensor) -> Tensor:
        return self.main_model.temporal_in_proj(z_encoded)

    def prealign(self, z_blended: Tensor, past_ticks: Tensor) -> Tensor:
        z = z_blended
        z = z + self.main_model.embed_delay(z, past_ticks)
        z = self.main_model.vit3d(z)
        z = einops.rearrange(z, "b c f h w -> b (c f) h w", f=4)
        return z

    def head(
        self,
        z_prealigned: Tensor,
        x_local_size: tuple[int, int],
        output_keys: Sequence[str] = (
            "downlink_features",
            # "downlink_seg_logits",
        ),
    ) -> dict[str, Tensor]:
        h_local, w_local = x_local_size
        output_size = (h_local // 8, w_local // 8)
        outputs = {}

        if "downlink_seg_logits" in output_keys:
            outputs["downlink_seg_logits"] = self.main_model.head(z_prealigned)

        if "downlink_features" in output_keys:
            z = self.mlp_pre_pool(z_prealigned)
            z = F.adaptive_avg_pool2d(z, output_size=output_size)
            z = self.mlp_post_pool(z)
            outputs["downlink_features"] = z

        return outputs


@register_model("dedelayed_v1_efficientvitl1_efficientvitb0_local")
class Dedelayed_v1_EfficientViTL1_EfficientViTB0_Local(Dedelayed_v1_Local):
    def __init__(
        self,
        name="efficientvit-seg-b0-cityscapes",
        *,
        normalization_src: ImageNormalizationKind = "01",
        normalization_dest: ImageNormalizationKind = "imagenet",
    ):
        from efficientvit.seg_model_zoo import create_efficientvit_seg_model

        super().__init__()
        self.normalization_src: ImageNormalizationKind = normalization_src
        self.normalization_dest: ImageNormalizationKind = normalization_dest
        self.image_model = create_efficientvit_seg_model(name, pretrained=True)

    def downlink_features_shape(
        self, x_local_size: tuple[int, int]
    ) -> tuple[int, int, int]:
        h_local, w_local = x_local_size
        return (32, h_local // 8, w_local // 8)

    def forward(self, x_local: Tensor, *, downlink_features: Tensor | None = None):
        from efficientvit.models.nn.ops import OpSequential
        from efficientvit.models.utils.list import list_sum

        model = self.image_model
        input_stem = cast(OpSequential, model.backbone.input_stem)
        backbone = model.backbone
        head = model.head
        z = x_local
        z = renormalize(
            z, src=self.normalization_src, dest=self.normalization_dest, channel_dim=1
        )

        # z = backbone(z)
        output_dict = {}
        output_dict["input"] = z
        z = input_stem(z)
        for i in range(4):
            output_dict[f"stage{i}"] = z
            if i == 2 and downlink_features is not None:
                z = z + downlink_features
            z = backbone.stages[i](z)
        output_dict["stage4"] = z
        output_dict["stage_final"] = z
        z = output_dict

        # z = head(z)
        feature_dict = z
        feat = [op(feature_dict[k]) for k, op in zip(head.input_keys, head.input_ops)]
        feat = list_sum(feat)
        assert head.post_input is None
        feat = head.middle(feat)
        for k, op in zip(head.output_keys, head.output_ops):
            feature_dict[k] = op(feat)
        z = feature_dict

        logits = z["segout"]

        return {
            "seg_logits": logits,
        }


@register_model("dedelayed_v1_efficientvitl1_efficientvitb0")
class Dedelayed_v1_EfficientViTL1_EfficientViTB0(Dedelayed_v1_Fused):
    def __init__(
        self,
        remote_model: Dedelayed_v1_EfficientViTL1_EfficientViTB0_Remote,
        local_model: Dedelayed_v1_EfficientViTL1_EfficientViTB0_Local,
        *,
        num_classes: int = 19,
    ):
        super().__init__()
        self.remote_model = remote_model
        self.local_model = local_model
        self.num_classes = num_classes
        self.drop_downlink_features_prob = 0.0

    def forward(
        self,
        x_local: Tensor,
        x_remote: Tensor,
        past_ticks: Tensor,
        local_only: bool = False,
    ):
        drop_downlink_features = self.training and (
            torch.rand((), device=x_local.device).item()
            < self.drop_downlink_features_prob
        ) or local_only

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
