import einops
import torch
from efficientvit.seg_model_zoo import create_efficientvit_seg_model

from dedelayed.layers.splitvid_v10 import (
    ConvNormActND,
    GELUTanh,
    GroupNorm,
    GroupNorm8,
    VitBlockND,
)


class EfficientViTSeg3D(torch.nn.Module):
    def __init__(
        self,
        name="efficientvit-seg-l1-cityscapes",
        middle=True,
        *,
        pretrained_image_model: bool = True,
        temporal_depth: int = 12,
        temporal_width: int = 256,
        temporal_expand_ratio: int = 4,
        temporal_norm_groups: int = 8,
    ):
        super().__init__()
        self.middle = middle
        self.image_model = create_efficientvit_seg_model(
            name, pretrained=pretrained_image_model
        )
        self.temporal_in_proj = (
            torch.nn.Identity()
            if temporal_width == 256
            else torch.nn.Conv3d(256, temporal_width, kernel_size=1, bias=False)
        )
        self.vit3d = torch.nn.Sequential(
            *[
                VitBlockND(
                    dim=3,
                    in_channels=temporal_width,
                    norm_layer=lambda num_channels: GroupNorm(
                        num_channels, num_groups=temporal_norm_groups
                    ),
                    act_layer=GELUTanh,
                    head_dim=32,
                    drop_path=0.1,
                    expand_ratio=temporal_expand_ratio,
                )
                for _ in range(temporal_depth)
            ]
        )
        self.head = torch.nn.Sequential(
            ConvNormActND(
                2,
                in_channels=4 * temporal_width,
                out_channels=256,
                norm_layer=GroupNorm8,
                act_layer=torch.nn.Identity,
                kernel_size=1,
            ),
            *self.image_model.head.middle.op_list,
            torch.nn.Conv2d(256, 19, kernel_size=1, stride=1),
        )
        self.learnable_delay_embedding = torch.nn.Sequential(
            ConvNormActND(
                dim=3,
                in_channels=1,
                out_channels=1024,
                norm_layer=GroupNorm8,
                act_layer=GELUTanh,
                kernel_size=1,
                bias=True,
            ),
            ConvNormActND(
                dim=3,
                in_channels=1024,
                out_channels=temporal_width,
                norm_layer=GroupNorm8,
                act_layer=torch.nn.Identity,
                kernel_size=1,
                bias=True,
            ),
        )

    def embed_delay(self, video_embedding, delay):
        delay_input = torch.as_tensor(
            delay, device=video_embedding.device, dtype=video_embedding.dtype
        ).reshape(-1, 1, 1, 1, 1)
        return self.learnable_delay_embedding(delay_input)

    def forward_images(self, x):
        B, C, F, H, W = x.shape
        x = {"-1": einops.rearrange(x, "b c f h w -> (b f) c h w")}
        for i_stage, stage in enumerate(self.image_model.backbone.stages):
            x[f"{i_stage}"] = stage(x[f"{i_stage - 1}"])
        prefix = "stage"
        for i_op, op in enumerate(self.image_model.head.input_ops):
            k_stage = self.image_model.head.input_keys[i_op][len(prefix) :]
            assert self.image_model.head.input_keys[i_op].startswith(prefix)
            if i_op == 0:
                y = op(x[k_stage])
            else:
                y += op(x[k_stage])
        if self.middle:
            y = self.image_model.head.middle(y)
        return einops.rearrange(y, "(b f) c h w -> b c f h w", f=F)

    def forward_features(self, x, delay):
        video_embedding = self.forward_images(x)
        video_embedding = self.temporal_in_proj(video_embedding)
        delay_embedding = self.embed_delay(video_embedding, delay)
        return self.vit3d(video_embedding + delay_embedding)

    def pool(self, x):
        B, C, F, H, W = x.shape
        return einops.rearrange(
            torch.nn.functional.adaptive_avg_pool3d(x, output_size=(4, H, W)),
            "b c f h w -> b (c f) h w",
            f=4,
        )

    def forward(self, x, delay):
        return self.head(self.pool(self.forward_features(x, delay)))
