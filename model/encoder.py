from typing import Tuple, Union
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from TeXOCR.model.attention import EncoderLayers
from TeXOCR.model.resnet import *

class PatchEmbedding(nn.Module):
    """Standard patch embedding layer for the ViT (https://arxiv.org/abs/2010.11929)."""
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Patchify input images with a convolutional layer
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    

class HybridEmbedding(nn.Module):
    """Hybrid patch embedding enables the use of CNN feature maps instead of raw image patches."""
    def __init__(
        self, 
        img_size: Union[int, Tuple[int, int]],
        patch_size: int, 
        in_channels: int, 
        embed_dim: int, 
        backbone_net: nn.Module
    ):
        super().__init__()

        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.backbone_net = backbone_net

        # Get feature size from backbone network
        with torch.no_grad():
            tr = backbone_net.training
            if tr:
                backbone_net.eval()
            out = self.backbone_net(torch.zeros(1, in_channels, *img_size))
            feat_size = out.shape[-2:]
            feat_dim = out.shape[1]
            backbone_net.train(tr)

        self.grid_size = (feat_size[0] // patch_size, feat_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Project to embedding dimension
        self.proj = nn.Conv2d(feat_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone_net(x)
        
        if isinstance(x, (tuple, list)):
            x = x[-1]

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]],
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        num_layers: int,
        heads: int,
        drop_rate: float = 0.,
        embed_layer: nn.Module = PatchEmbedding,
        norm_layer: nn.Module = nn.LayerNorm,
        ff_kwargs: dict = {}
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_tokens = 1 # Single class token
        self.height, self.width = img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size)
        self.patch_size = patch_size

        # Patch embedding layer
        self.patch_embed = embed_layer(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.dropout = nn.Dropout(p=drop_rate)

        # Transformer encoder layers
        self.attn_layers = EncoderLayers(
            embed_dim=embed_dim,
            num_layers=num_layers,
            heads=heads,
            cross_attend=False,
            **ff_kwargs
        )

        # Classification head (if num_classes > 0)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.norm = norm_layer(embed_dim)

        # self.init_weights(weight_init)

    # def init_weights(self, weight_init):
    #     pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Get patch embeddings and concatenate class token
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Get indices for positional embeddings even if x is not size of max dimensions
        h, w = H // self.patch_size, W // self.patch_size
        max_h, max_w = self.height // self.patch_size, self.width // self.patch_size
        grid_indices = torch.arange(max_h * max_w).reshape(max_h, max_w)
        pos_ids = grid_indices[:h, :w].reshape(-1)
        pos_ids = torch.cat((torch.zeros(1), pos_ids + 1), dim=0).long()
        
        x += self.pos_embed[:, pos_ids]
        x = self.dropout(x)
        
        # Run transformer blocks
        x, intermediates = self.attn_layers(x, mask=None, return_hidden=True)
        x = self.norm(x)

        # Classification head (if num_classes > 0)
        x = self.head(x)
        return x
    

class VisionEncoder(VisionTransformer):
    """Vision encoder for TeXOCR."""
    def __init__(self, **kwargs):
        assert 'num_classes' not in kwargs, 'VisionEncoder does not require num_classes.'
        super().__init__(num_classes=0, **kwargs)


class HybridEmbedResNet(HybridEmbedding):
    """Hybrid embedding module with ResNet backbone net."""
    def __init__(self, **kwargs):
        backbone_net = kwargs.get('backbone_net', ResNetV2(depths=[2,4,6], in_channels=1))
        min_patch_size = 2 ** (len(backbone_net.depths) + 1)
        full_patch_size = kwargs.pop('patch_size', 16)
        reduced_patch_size = full_patch_size // min_patch_size
        super().__init__(patch_size=reduced_patch_size, **kwargs)


def create_encoder(config: dict):
    """Create a VisionEncoder from a config dict."""
    patch_size = config['patch_size']
    encoder_args = config['encoder']

    backbone_net = ResNetV2(
        depths=[2, 4, 6],
        in_channels=1
    )

    encoder = VisionEncoder(
        img_size=(160, 1008),
        patch_size=patch_size,
        in_channels=encoder_args['n_channels'],
        embed_dim=config['embed_dim'],
        num_layers=encoder_args['num_layers'],
        heads=encoder_args['heads'],
        embed_layer=partial(HybridEmbedResNet, backbone_net=backbone_net),
    )
    return encoder

# if __name__ == '__main__':

#     model = VisionTransformer(
#         img_size=224,
#         patch_size=16,
#         in_channels=1,
#         num_classes=0,
#         embed_dim=256,
#         depth=6,
#         num_heads=8
#     )

#     print(model(torch.randn(1,1,224,224)).shape)