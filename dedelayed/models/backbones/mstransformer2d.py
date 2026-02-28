# Copyright (c) 2026, InterDigital
# All rights reserved.
# See LICENSE under the root folder.

import torch
from timm.models.efficientvit_mit import GELUTanh

from dedelayed.layers.splitvid_v10 import (
    AnalysisConvND,
    ConvNormActND,
    DSGSConvND,
    GroupNorm8,
    MBConvND_head,
    Residual,
    SynthesisConvND,
    VitBlockND,
)


class MSTransformer2D(torch.nn.Module):
    def __init__(self, cls_classes, seg_classes):
        super().__init__()
        self.T1 = torch.nn.Sequential(
            AnalysisConvND(dim=2, in_ch=3, out_ch=64, stride=4, groups=1),
            torch.nn.Sequential(
                *[
                    Residual(DSGSConvND(dim=2, in_channels=64, groups=2), d=0.0)
                    for _ in range(1)
                ]
            ),
            AnalysisConvND(dim=2, in_ch=64, out_ch=96, stride=2, groups=1),
            torch.nn.Sequential(
                *[
                    Residual(DSGSConvND(dim=2, in_channels=96, groups=3), d=1e-2)
                    for _ in range(2)
                ]
            ),
        )
        self.T2 = torch.nn.Sequential(
            AnalysisConvND(dim=2, in_ch=96, out_ch=192, stride=2, groups=1),
            torch.nn.Sequential(
                *[
                    Residual(DSGSConvND(dim=2, in_channels=192, groups=3), d=2e-2)
                    for _ in range(2)
                ]
            ),
            torch.nn.Sequential(
                *[
                    VitBlockND(
                        dim=2,
                        in_channels=192,
                        norm_layer=GroupNorm8,
                        act_layer=GELUTanh,
                        head_dim=16,
                        expand_ratio=4,
                        drop_path=3e-2,
                    )
                    for _ in range(3)
                ]
            ),
        )
        self.P2 = torch.nn.Sequential(
            SynthesisConvND(dim=2, in_ch=192, out_ch=96, stride=2)
        )
        self.T3 = torch.nn.Sequential(
            AnalysisConvND(dim=2, in_ch=192, out_ch=320, stride=2, groups=1),
            torch.nn.Sequential(
                *[
                    VitBlockND(
                        dim=2,
                        in_channels=320,
                        norm_layer=GroupNorm8,
                        act_layer=GELUTanh,
                        head_dim=32,
                        expand_ratio=4,
                        drop_path=4e-2,
                    )
                    for _ in range(4)
                ]
            ),
        )
        self.P3 = torch.nn.Sequential(
            SynthesisConvND(dim=2, in_ch=320, out_ch=96, stride=4)
        )
        self.seg_head = torch.nn.Sequential(
            torch.nn.Sequential(
                *[
                    Residual(
                        MBConvND_head(
                            dim=2,
                            in_channels=96,
                            norm_layer=GroupNorm8,
                            act_layer=GELUTanh,
                            expand_ratio=3,
                        ),
                        d=0.0,
                    )
                    for _ in range(3)
                ]
            ),
            torch.nn.Conv2d(96, seg_classes, kernel_size=1, stride=1, bias=True),
        )
        self.cls_head = torch.nn.Sequential(
            ConvNormActND(
                dim=2,
                in_channels=320,
                out_channels=3072,
                kernel_size=1,
                norm_layer=GroupNorm8,
                act_layer=GELUTanh,
            ),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Conv2d(3072, 3200, kernel_size=1, stride=1, bias=False),
            torch.nn.GroupNorm(1, 3200, eps=1e-7),
            GELUTanh(),
            torch.nn.Conv2d(3200, cls_classes, kernel_size=1, stride=1, bias=True),
        )

    def forward_features(self, x):
        x = self.T1(x)
        x = self.T2(x)
        x = self.T3(x)
        return x

    # def forward_seg(self, x):
    #     y = x = self.T1(x)
    #     x = self.T2(x)
    #     y += self.P2(x)
    #     x = self.T3(x)
    #     y += self.P3(x)
    #     return self.seg_head(y)

    def forward_seg(self, x):
        y = x = self.T1(x)
        x = self.T2(x)
        y = y + self.P2(x)
        x = self.T3(x)
        y = y + self.P3(x)
        return self.seg_head(y)

    def forward_cls(self, x):
        x = self.T1(x)
        x = self.T2(x)
        x = self.T3(x)
        return self.cls_head(x)[:, :, 0, 0]
