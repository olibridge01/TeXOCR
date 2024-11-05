import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import VisionEncoder
from model.decoder import TransformerDecoder

class OCRModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, src_pad_idx: int, trg_pad_idx: int, device: torch.device):
        super(OCRModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Source tensor.
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trg: Target tensor.
        """
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Source tensor.
            trg: Target tensor.
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, mask=None)
        out = self.decoder(trg, enc_src, src_mask=None, trg_mask=trg_mask)
        return out
    
    def generate(self, src: torch.Tensor) -> torch.Tensor:
        pass
    
if __name__ == "__main__":
    

    patch_size = 16
    n_channels = 3
    batch_size = 64
    trg_vocab_size = 256
    src_pad_idx = 0
    trg_pad_idx = 0
    embed_dim = 512
    num_layers = 6
    exp_factor = 4
    heads = 8
    dropout = 0.
    device = torch.device("cpu")
    max_length = 100

    model = OCRModel(
        patch_size,
        n_channels,
        batch_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_dim,
        num_layers,
        exp_factor,
        heads,
        dropout,
        device,
        max_length
    )

    test_img = torch.randn((64, 3, 224, 224))
    test_target = torch.randint(0, 256, (64, 42))

    out = model(test_img, test_target)

    print(out.shape)

    