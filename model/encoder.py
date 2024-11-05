import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from model.decoder import TransformerBlock

class InputEmbedding(nn.Module):
    """Embedding layer for Vision Transformer encoder. Patchifies the input image and projects them into the latent space."""
    def __init__(self, patch_size: int, n_channels: int, embed_dim: int, batch_size: int, device: torch.device):
        """
        Args:
            patch_size: Size of the patch.
            n_channels: Number of channels in the input image.
            latent_size: Size of the latent space.
            batch_size: Size of the batch.
            device: Device to run the model on.
        """
        super(InputEmbedding, self).__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.device = device

        self.input_size = self.patch_size * self.patch_size * self.n_channels

        # Linear projection
        self.linear_proj = nn.Linear(self.input_size, self.embed_dim)

        # Class token
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.embed_dim).to(self.device))

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.embed_dim).to(self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor.
        """
        x = x.to(self.device)

        # Patchify input images
        patches = einops.rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        linear_proj = self.linear_proj(patches).to(self.device)
        b, n, _ = linear_proj.shape

        linear_proj = torch.cat((self.class_token, linear_proj), dim=1)
        pos_embedding = einops.repeat(self.pos_embedding, 'b 1 d -> b n d', n=n+1)

        linear_proj += pos_embedding

        return linear_proj
    

class VisionEncoder(nn.Module):
    """Vision Transformer (ViT) encoder module."""
    def __init__(
            self,
            patch_size: int,
            n_channels: int,
            embed_dim: int,
            num_layers: int,
            heads: int,
            device: torch.device,
            exp_factor: int,
            dropout: float,
            batch_size: int
    ):
        """
        Args:
            patch_size: Patch size.
            n_channels: Number of channels.
            embed_dim: Embedding dimension.
            num_layers: Number of layers.
            heads: Number of attention heads.
            device: Device.
            exp_factor: MLP dimension expansion factor.
            dropout: Dropout rate.
            batch_size: Batch size.
        """
        super(VisionEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.device = device

        self.input_embedding = InputEmbedding(patch_size, n_channels, embed_dim, batch_size=batch_size, device=device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    heads,
                    dropout=dropout,
                    exp_factor=exp_factor,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor.
            mask: Mask tensor.
        """
        N, C, H, W = x.shape # (batch_size, channels, height, width)

        # Patchify input images and get patch embeddings
        x = self.input_embedding(x) # (batch_size, num_patches, embed_dim)
        out = self.dropout(x)

        for layer in self.layers:
            out = layer(out, out, out, mask) # (batch_size, num_patches, embed_dim)

        return out
    

if __name__ == '__main__':

    patch_size = 16
    latent_size = 768
    n_channels = 3
    num_heads = 12
    num_encoders = 12
    dropout = 0.1
    num_classes = 10
    size = 224

    epochs = 10
    base_lr = 1e-2
    weight_decay = 0.03
    batch_size = 4

    test_input = torch.randn((4, 3, 224, 224))
    test_class = VisionEncoder(patch_size, n_channels, latent_size, num_encoders, num_heads, torch.device('cpu'), 4, dropout, batch_size)

    out = test_class(test_input)
    print(out.shape)