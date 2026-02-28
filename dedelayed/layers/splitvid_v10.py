# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

import math
from typing import Optional

import torch
from timm.layers import drop_path, use_fused_attn
from timm.models.efficientvit_mit import GELUTanh


class GroupNorm(torch.nn.Module):
    def __init__(self, num_features, num_groups=8, eps=1e-7, affine=True):
        super().__init__()
        self.groupnorm = torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=num_features, eps=eps, affine=affine
        )

    def forward(self, x):
        return self.groupnorm(x)


class GroupNorm8(torch.nn.Module):
    def __init__(self, num_features, eps=1e-7, affine=True):
        super().__init__()
        self.groupnorm = torch.nn.GroupNorm(
            num_groups=8, num_channels=num_features, eps=eps, affine=affine
        )

    def forward(self, x):
        return self.groupnorm(x)


class ConvNormActND(torch.nn.Module):
    def __init__(
        self,
        dim,
        in_channels,
        out_channels,
        norm_layer,
        act_layer,
        kernel_size=3,
        stride=1,
        groups=1,
        bias=False,
        padding=None,
    ):
        super().__init__()
        Conv = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d][dim - 1]
        if padding is None:
            padding = kernel_size // 2
        self.conv = Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvTransposeNormActND(torch.nn.Module):
    def __init__(
        self,
        dim,
        in_channels,
        out_channels,
        norm_layer,
        act_layer,
        kernel_size=3,
        stride=1,
        groups=1,
        bias=False,
        padding=None,
    ):
        super().__init__()
        Conv = [
            torch.nn.ConvTranspose1d,
            torch.nn.ConvTranspose2d,
            torch.nn.ConvTranspose3d,
        ][dim - 1]
        if padding is None:
            padding = kernel_size // 2
        self.conv = Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MBConvND(torch.nn.Module):
    def __init__(self, dim, in_channels, norm_layer, act_layer, expand_ratio=4):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        self.inverted_conv = ConvNormActND(
            dim,
            in_channels,
            mid_channels,
            kernel_size=1,
            norm_layer=torch.nn.Identity,
            act_layer=act_layer,
            bias=True,
        )
        self.depth_conv = ConvNormActND(
            dim,
            mid_channels,
            mid_channels,
            kernel_size=3,
            groups=mid_channels,
            norm_layer=torch.nn.Identity,
            act_layer=act_layer,
            bias=True,
        )
        self.point_conv = ConvNormActND(
            dim,
            mid_channels,
            in_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            act_layer=torch.nn.Identity,
            bias=False,
        )

    def forward(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConvND_head(torch.nn.Module):
    def __init__(self, dim, in_channels, norm_layer, act_layer, expand_ratio=4):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        self.inverted_conv = ConvNormActND(
            dim,
            in_channels,
            mid_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            act_layer=act_layer,
            bias=True,
        )
        self.depth_conv = ConvNormActND(
            dim,
            mid_channels,
            mid_channels,
            kernel_size=3,
            groups=mid_channels,
            norm_layer=norm_layer,
            act_layer=act_layer,
            bias=True,
        )
        self.point_conv = ConvNormActND(
            dim,
            mid_channels,
            in_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            act_layer=torch.nn.Identity,
            bias=False,
        )

    def forward(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class AnalysisConvND(torch.nn.Module):
    def __init__(
        self,
        dim,
        in_ch,
        out_ch,
        stride=2,
        groups=1,
        norm_layer=GroupNorm8,
        act_layer=torch.nn.Identity,
    ):
        super().__init__()
        self.ops = ConvNormActND(
            dim=dim,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=stride,
            stride=stride,
            padding=0,
            groups=groups,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    def forward(self, x):
        return self.ops(x)


class SynthesisConvND(torch.nn.Module):
    def __init__(
        self,
        dim,
        in_ch,
        out_ch,
        stride=2,
        groups=1,
        norm_layer=GroupNorm8,
        act_layer=torch.nn.Identity,
    ):
        super().__init__()
        self.ops = ConvTransposeNormActND(
            dim=dim,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=stride,
            stride=stride,
            padding=0,
            groups=groups,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    def forward(self, x):
        return self.ops(x)


class DSGSConvND(torch.nn.Module):
    def __init__(
        self,
        dim,
        in_channels,
        groups,
        norm_layer=GroupNorm8,
        act_layer=GELUTanh,
        expand_ratio=2,
    ):
        super().__init__()
        self.ops = torch.nn.Sequential(
            ConvNormActND(
                dim=dim,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                groups=groups,
                norm_layer=torch.nn.Identity,
                act_layer=torch.nn.Identity,
            ),
            ConvNormActND(
                dim=dim,
                in_channels=in_channels,
                out_channels=expand_ratio * in_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ),
            ConvNormActND(
                dim=dim,
                in_channels=expand_ratio * in_channels,
                out_channels=in_channels,
                kernel_size=1,
                norm_layer=torch.nn.Identity,
                act_layer=torch.nn.Identity,
            ),
        )

    def forward(self, x):
        return self.ops(x)


class Residual(torch.nn.Module):
    def __init__(self, main, d):
        super().__init__()
        self.main = main
        self.drop_path = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + drop_path(
            self.main(x),
            drop_prob=self.drop_path,
            training=self.training,
            scale_by_keep=False,
        )


class AttentionND(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        embed_dim: int,
        dim_out: Optional[int] = None,
        dim_head: int = 32,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim in (1, 2, 3), "`dim` must be 1, 2 or 3"
        assert use_fused_attn(), "no support for fused attention"
        Conv = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d][dim - 1]
        dim_out = dim_out or embed_dim
        dim_attn = dim_out
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.qkv = Conv(embed_dim, dim_attn * 3, 1, bias=True)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = Conv(dim_attn, dim_out, 1, bias=True)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, *spatial_shape = x.shape
        N = math.prod(spatial_shape)
        q, k, v = (
            self.qkv(x)
            .view(B, self.num_heads, -1, self.dim_head)
            .reshape(B, self.num_heads, -1, 3, self.dim_head)
            .unbind(3)
        )
        # if self.slow_gqa:
        #     hq, hk, hv = q.size(1), k.size(1), v.size(1)
        #     k = k.repeat_interleave(hq // hk, dim=1)
        #     v = v.repeat_interleave(hq // hv, dim=1)
        x = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(-1, -2),
            k.transpose(-1, -2),
            v.transpose(-1, -2),
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            # enable_gqa=True,  # NOTE: This has no effect when hq == hk == hv.
        )
        x = x.transpose(-1, -2).reshape(B, -1, *spatial_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VitBlockND(torch.nn.Module):
    def __init__(
        self, dim, in_channels, norm_layer, act_layer, head_dim, expand_ratio, drop_path
    ):
        super().__init__()
        self.context_module = Residual(
            AttentionND(dim, in_channels, in_channels, dim_head=head_dim), d=drop_path
        )
        self.local_module = Residual(
            MBConvND(
                dim,
                in_channels,
                norm_layer=norm_layer,
                act_layer=act_layer,
                expand_ratio=expand_ratio,
            ),
            d=drop_path,
        )

    def forward(self, x):
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class PrepoolBlock(torch.nn.Sequential):
    def __init__(self, in_channels: int = 1024, channels: int = 256):
        super().__init__(
            ConvNormActND(
                2,
                in_channels=in_channels,
                out_channels=channels,
                norm_layer=GroupNorm8,
                act_layer=GELUTanh,
                kernel_size=1,
            ),
            torch.nn.Sequential(
                *[
                    Residual(
                        MBConvND_head(
                            dim=2,
                            in_channels=channels,
                            norm_layer=GroupNorm8,
                            act_layer=GELUTanh,
                            expand_ratio=2,
                        ),
                        d=0.0,
                    )
                    for _ in range(3)
                ]
            ),
        )


class PostpoolBlock(torch.nn.Sequential):
    def __init__(self, channels: int = 256, out_channels: int = 96):
        super().__init__(
            torch.nn.Sequential(
                *[
                    Residual(
                        MBConvND_head(
                            dim=2,
                            in_channels=channels,
                            norm_layer=GroupNorm8,
                            act_layer=GELUTanh,
                            expand_ratio=2,
                        ),
                        d=0.0,
                    )
                    for _ in range(3)
                ]
            ),
            torch.nn.Conv2d(channels, out_channels, kernel_size=1, stride=1),
        )
