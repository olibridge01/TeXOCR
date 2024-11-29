# ResNet code adapted from timm
import math
from functools import partial
from typing import List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from TeXOCR.utils import padding_val, pad_same


class GroupNormAct(nn.GroupNorm):
    """GroupNorm + ReLU activation for use within ResNet."""
    def __init__(
        self, 
        num_channels: int, 
        num_groups: int = 32, 
        eps: float = 1e-5, 
        affine: bool = True, 
        act: bool = True,
        act_layer: nn.Module = nn.ReLU
    ):
        super(GroupNormAct, self).__init__(num_groups, num_channels, eps=eps, affine=affine)
        if act:
            args = dict(inplace=True)
            self.act = act_layer(**args)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.act(x)
        return x
    

class StdConv2d(nn.Conv2d):
    """Conv2d with weight standardisation (https://arxiv.org/abs/2101.08692) and SAME padding."""
    def __init__(
            self,
            in_channel: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: str = 'SAME',
            dilation: int = 1,
            groups: int = 1,
            bias: bool = False,
            eps: float = 1e-6
    ):
        padding, dynamic = padding_val(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(in_channel, out_channels, kernel_size, stride=stride, padding=padding, 
                         dilation=dilation, groups=groups, bias=bias)
        self.eps = eps
        self.same = dynamic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.same:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps
        ).reshape_as(self.weight)
        x = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class MaxPool2d(nn.MaxPool2d):
    """Max pooling with SAME padding."""
    def __init__(self, kernel_size: int, stride=None, padding=0, dilation=1, ceil_mode=False):
        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        dilation = (dilation, dilation)
        super(MaxPool2d, self).__init__(kernel_size, stride, (0, 0), dilation, ceil_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pad_same(x, self.kernel_size, self.stride, value=-float('inf'))
        return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
    

class DownSample(nn.Module):
    """Downsample convolutional layer."""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        conv_layer: nn.Module = StdConv2d, 
        norm_layer: nn.Module = GroupNormAct
    ):
        super(DownSample, self).__init__()
        self.conv = conv_layer(in_channels, out_channels, kernel_size=1, stride=stride)
        self.norm = norm_layer(out_channels, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))
    

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet."""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int = None, 
        bottle_ratio: float = 0.25, 
        stride: int = 1, 
        dilation: int = 1, 
        first_dilation: int = None, 
        groups: int = 1,
        proj_layer: nn.Module = None, 
    ):
        super().__init__()
        first_dilation = first_dilation or dilation
        out_channels = out_channels or in_channels
        mid_channels = int(out_channels * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_channels, 
                out_channels, 
                stride=stride,
                conv_layer=StdConv2d, 
                norm_layer=GroupNormAct
            )
        else:
            self.downsample = None

        # Meat of the bottleneck block
        self.block_list = nn.ModuleList([
            StdConv2d(in_channels, mid_channels, kernel_size=1),
            GroupNormAct(mid_channels),
            StdConv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups),
            GroupNormAct(mid_channels),
            StdConv2d(mid_channels, out_channels, kernel_size=1),
            GroupNormAct(out_channels, act=False),
        ])
        self.act = nn.ReLU(inplace=True)
        self.block = nn.Sequential(*self.block_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle downsampling and residual connection
        res = x
        if self.downsample is not None:
            res = self.downsample(x)
        
        x = self.block(x)
        x = self.act(x + res)
        return x


class Stage(nn.Module):
    """ResNet Stage."""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int, 
        dilation: int, 
        depth: int, 
        bottle_ratio: float = 0.25, 
        groups: int = 1,
    ):
        super(Stage, self).__init__()
        if dilation in (1, 2):
            first_dilation = 1
        else:
            first_dilation = 2

        # Set proj layer for first block
        proj_layer = DownSample
        prev_channels = in_channels

        self.stage_blocks = nn.Sequential()

        # Create blocks for stage
        for id in range(depth):
            stride = stride if id == 0 else 1
            self.stage_blocks.add_module(
                str(id), 
                Bottleneck(
                    prev_channels, 
                    out_channels, 
                    stride=stride, 
                    dilation=dilation,
                    bottle_ratio=bottle_ratio, 
                    groups=groups,
                    first_dilation=first_dilation, 
                    proj_layer=proj_layer, 
                )
            )
            prev_channels = out_channels
            first_dilation = dilation
            proj_layer = None # Remaining blocks have no proj layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage_blocks(x)


class ResNetV2(nn.Module):
    """ResNet-V2 model (https://arxiv.org/abs/1603.05027)."""
    def __init__(
        self,
        depths: List[int],
        channels: List[int] = [256, 512, 1024, 2048], 
        num_classes: int = 0,
        in_channels: int = 1,
        stem_channels: int = 64,
        out_stride: int = 32,
        conv_layer: nn.Module = StdConv2d,
        norm_layer: nn.Module = GroupNormAct,
    ):
        super().__init__()
        self.depths = depths
        self.num_classes = num_classes

        # Create ResNet stem
        self.stem = nn.Sequential(
            conv_layer(in_channels, stem_channels, kernel_size=7, stride=2),
            norm_layer(stem_channels),
            MaxPool2d(kernel_size=3, stride=2)
        )

        prev_channels = stem_channels
        curr_stride = 4
        dilation = 1

        self.stages = nn.Sequential()
        for stage_id, (d, c) in enumerate(zip(depths, channels)):

            out_channels = c
            stride = 1 if stage_id == 0 else 2
            if curr_stride >= out_stride:
                dilation *= stride
                stride = 1
            
            # Create ResNet stage
            stage = Stage(
                prev_channels,
                out_channels,
                stride=stride,
                dilation=dilation,
                depth=d,
            )
            prev_channels = out_channels
            curr_stride *= stride
            self.stages.add_module(str(stage_id), stage)

        self.num_features = prev_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return x
    
# if __name__ == '__main__':
#     model = ResNetV2([2, 3, 7])
#     print(model(torch.randn(1, 3, 224, 224)).shape)