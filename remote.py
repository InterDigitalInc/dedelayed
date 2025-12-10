# This code is provided solely as supplementary material for the anonymized review of
# CVPR 2025 submission REDACTED
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
from local import VitBlockND, GroupNorm8, GELUTanh, ConvNormActND

class RemoteModel(torch.nn.Module):
    def __init__(self, name='efficientvit-seg-l1-cityscapes', middle=True):
        super().__init__()
        self.middle = middle
        self.image_model = create_efficientvit_seg_model(name, pretrained=False)
        self.vit3d = torch.nn.Sequential(*[
                VitBlockND(dim=3, in_channels=256, norm_layer=GroupNorm8, act_layer=GELUTanh, head_dim=32, drop_path=0.1, expand_ratio=4)
                for _ in range(12)])
        self.head = torch.nn.Sequential(
            ConvNormActND(2, in_channels=1024, out_channels=256, norm_layer=GroupNorm8, act_layer=torch.nn.Identity, kernel_size=1),
            *self.image_model.head.middle.op_list,
            torch.nn.Conv2d(256, 19, kernel_size=1, stride=1)            
        )
        self.learnable_delay_embedding = torch.nn.Sequential(
            ConvNormActND(
                dim=3,
                in_channels = 1,
                out_channels = 1024,
                norm_layer = GroupNorm8,
                act_layer = GELUTanh,
                kernel_size=1,
                bias = True
            ),
            ConvNormActND(
                dim=3,
                in_channels = 1024,
                out_channels = 256,
                norm_layer = GroupNorm8,
                act_layer = torch.nn.Identity,
                kernel_size=1,
                bias=True,
            ),
        )

    def embed_delay(self, video_embedding, delay):
        b,_,f,h,w = video_embedding.shape
        return self.learnable_delay_embedding(
            delay*torch.ones(b,1,f,h,w,device=video_embedding.device)
        )        
        
    def forward_images(self,x):
        B, C, F, H, W = x.shape
        x = {'-1': einops.rearrange(x, 'b c f h w -> (b f) c h w')}
        for i_stage, stage in enumerate(self.image_model.backbone.stages):
            x[f'{i_stage}'] = stage(x[f'{i_stage-1}'])
        for i_op, op in enumerate(self.image_model.head.input_ops):
            i_stage = self.image_model.head.input_keys[i_op][5:]
            if i_op == 0:
                y = op(x[i_stage])
            else:
                y += op(x[i_stage])
        if self.middle:
            y = self.image_model.head.middle(y)
        return einops.rearrange(y, '(b f) c h w -> b c f h w',f=F)
    
    def forward_features(self,x,delay):
        video_embedding = self.forward_images(x)
        delay_embedding = self.embed_delay(video_embedding,delay)
        return self.vit3d(video_embedding+delay_embedding)

    def pool(self,x):
        B, C, F, H, W = x.shape
        return einops.rearrange(torch.nn.functional.adaptive_avg_pool3d(x, output_size=(4, H, W)), 'b c f h w -> b (c f) h w', f=4)

    def forward(self,x,delay):
        return self.head( self.pool( self.forward_features(x,delay) ) )