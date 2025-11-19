# This code is provided solely as supplementary material for the anonymized review of
# CVPR 2025 submission 13893
#
# It is intended exclusively for the purpose of verifying the reproducibility of the
# results reported in the submission during the double-blind peer review process.
#
# Distribution, reproduction, modification, or any other use beyond the CVPR 2025
# review process is strictly prohibited without prior written permission from the
# copyright holder.
#
# All intellectual property rights are reserved by the copyright owner.
# No license is granted. Provided "as is" without warranty of any kind, express or implied,
# including but not limited to warranties of merchantability, fitness for a particular purpose, # or non-infringement.
#
# For the anonymized review process only.

import torch, einops
from efficientvit.seg_model_zoo import create_efficientvit_seg_model
from local import GroupNorm8, GELUTanh, Residual, ConvNormActND, MBConvND_head, MSTransformer2D
from remote import RemoteModel

class FusedModel(torch.nn.Module):
    def __init__(self, cls_classes=1000, seg_classes=19):
        super().__init__()
        self.local_model = MSTransformer2D(cls_classes=cls_classes, seg_classes=seg_classes)
        self.remote_model = RemoteModel()
        self.mlp_pre_pool = torch.nn.Sequential(
            ConvNormActND(2, in_channels=1024, out_channels=256, norm_layer=GroupNorm8, act_layer=GELUTanh, kernel_size=1),
            torch.nn.Sequential(*[
                Residual(MBConvND_head(dim=2, in_channels=256, norm_layer=GroupNorm8, act_layer=GELUTanh, expand_ratio=2), d=0.0)
                for _ in range(3)]),
        )
        self.mlp_post_pool = torch.nn.Sequential(
            torch.nn.Sequential(*[
                Residual(MBConvND_head(dim=2, in_channels=256, norm_layer=GroupNorm8, act_layer=GELUTanh, expand_ratio=2), d=0.0)
                for _ in range(3)]),
            torch.nn.Conv2d(256, 96, kernel_size=1, stride=1)    
        )
        
    def forward(self, x_local, x_remote, delay):
        x = self.local_model.T1(x_local)
        _, _, H, W = x.shape

        x_remote = self.remote_model.forward_features(x_remote, delay)
        x_remote = einops.rearrange(x_remote, 'b c f h w -> b (c f) h w', f=4)
        x_remote = self.mlp_pre_pool(x_remote)
        x_remote = torch.nn.functional.adaptive_avg_pool2d(x_remote, output_size=(H, W))
        x_remote = self.mlp_post_pool(x_remote)
        y = x = x + x_remote
        
        x = self.local_model.T2(x)
        y = y + self.local_model.P2(x)
        x = self.local_model.T3(x)
        y = y + self.local_model.P3(x)
        return self.local_model.seg_head(y)