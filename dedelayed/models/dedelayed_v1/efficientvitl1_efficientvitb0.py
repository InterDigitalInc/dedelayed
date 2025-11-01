from __future__ import annotations

from typing import cast

import einops
import torch.nn as nn
import torch.nn.functional as F
from efficientvit.models.nn.ops import OpSequential
from efficientvit.models.utils.list import list_sum
from efficientvit.seg_model_zoo import create_efficientvit_seg_model
from torch import Tensor

from dedelayed.data.normalization import ImageNormalizationKind, renormalize
from dedelayed.layers.splitvid_v10 import PostpoolBlock, PrepoolBlock
from dedelayed.models.backbones.evit_vd import EfficientViTSeg3D
from dedelayed.registry import register_model


@register_model("dedelayed_v1_efficientvitl1_efficientvitb0_remote")
class Dedelayed_v1_EfficientViTL1_EfficientViTB0_Remote(nn.Module):
    def __init__(
        self,
        name="efficientvit-seg-l1-cityscapes",
        *,
        normalization_src: ImageNormalizationKind = "01",
        temporal_depth: int = 4,
        temporal_width: int = 96,
        temporal_expand_ratio: int = 1,
        temporal_norm_groups: int = 32,
        send_channels: int = 32,
    ):
        super().__init__()
        self.normalization_src: ImageNormalizationKind = normalization_src
        self.main_model = EfficientViTSeg3D(
            name=name,
            temporal_depth=temporal_depth,
            temporal_width=temporal_width,
            temporal_expand_ratio=temporal_expand_ratio,
            temporal_norm_groups=temporal_norm_groups,
            pretrained_image_model=True,
        )
        self.mlp_pre_pool = PrepoolBlock(in_channels=temporal_width * 4)
        self.mlp_post_pool = PostpoolBlock(out_channels=send_channels)

    def forward(
        self,
        x_remote: Tensor | None = None,
        *,
        z_remote: Tensor | None = None,
        x_local_size: tuple[int, int],
        past_ticks: int = 0,
    ):
        H_local, W_local = x_local_size
        output_size = (H_local // 8, W_local // 8)

        if z_remote is None:
            assert x_remote is not None
            z_remote = self.contextualize(x_remote)
        z = z_remote
        z = self.main_model.temporal_in_proj(z)
        z = z + self.main_model.embed_delay(z, past_ticks)
        z = self.main_model.vit3d(z)
        z = einops.rearrange(z, "b c f h w -> b (c f) h w", f=4)

        # send_seg_logits = self.main_model.head(z)

        z = self.mlp_pre_pool(z)
        z = F.adaptive_avg_pool2d(z, output_size=output_size)
        z = self.mlp_post_pool(z)
        send_features = {"stage2_backbone": z}

        return {
            "send": {
                "features": send_features,
                # "seg_logits": send_seg_logits,
            },
        }

    def contextualize(self, x: Tensor) -> Tensor:
        x = renormalize(x, src=self.normalization_src, dest="imagenet", channel_dim=1)
        return self.main_model.forward_images(x)

    def encode_image(self, x: Tensor) -> Tensor:
        z = einops.rearrange(x, "b c h w -> b c 1 h w")
        z = self.contextualize(z)
        z = einops.rearrange(z, "b c 1 h w -> b c h w")
        return z


@register_model("dedelayed_v1_efficientvitl1_efficientvitb0_local")
class Dedelayed_v1_EfficientViTL1_EfficientViTB0_Local(nn.Module):
    def __init__(
        self,
        name="efficientvit-seg-b0-cityscapes",
        *,
        normalization_src: ImageNormalizationKind = "01",
    ):
        super().__init__()
        self.normalization_src: ImageNormalizationKind = normalization_src
        self.image_model = create_efficientvit_seg_model(name, pretrained=True)

    def forward(self, x_local: Tensor, *, recv_stage2_backbone: Tensor | None = None):
        model = self.image_model
        input_stem = cast(OpSequential, model.backbone.input_stem)
        backbone = model.backbone
        head = model.head
        z = renormalize(
            x_local, src=self.normalization_src, dest="imagenet", channel_dim=1
        )

        # z = backbone(z)
        output_dict = {}
        output_dict["input"] = z
        z = input_stem(z)
        for i in range(4):
            output_dict[f"stage{i}"] = z
            if i == 2 and recv_stage2_backbone is not None:
                z = z + recv_stage2_backbone
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
class Dedelayed_v1_EfficientViTL1_EfficientViTB0(nn.Module):
    def __init__(
        self,
        remote_model: Dedelayed_v1_EfficientViTL1_EfficientViTB0_Remote,
        local_model: Dedelayed_v1_EfficientViTL1_EfficientViTB0_Local,
    ):
        super().__init__()
        self.remote_model = remote_model
        self.local_model = local_model

    def forward(self, x_local: Tensor, x_remote: Tensor, past_ticks: int = 0):
        out_remote = self.remote_model(
            x_remote,
            past_ticks=past_ticks,
            x_local_size=x_local.shape[-2:],
        )

        out_local = self.local_model(
            x_local,
            recv_stage2_backbone=out_remote["send"]["features"]["stage2_backbone"],
        )

        seg_logits = out_local["seg_logits"]

        return {
            "seg_logits": seg_logits,
        }
