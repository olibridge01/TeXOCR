import torch
import torch.nn as nn
import torch.nn.functional as F

from TeXOCR.model.encoder import create_encoder
from TeXOCR.model.decoder import create_decoder

class OCRModel(nn.Module):
    """TeXOCR model for image-to-LaTeX conversion."""
    def __init__(
        self, 
        encoder: nn.Module, 
        decoder: nn.Module, 
        bos_token: int,
        eos_token: int,
        trg_pad_idx: int, 
        device: torch.device
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        trg_mask = (trg != self.trg_pad_idx) # Shape (N, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src: torch.Tensor, trg: torch.Tensor, return_out: bool = False) -> torch.Tensor:
        # Create target mask
        trg_mask = self.make_trg_mask(trg)

        # Get encoder output and pass to decoder
        enc = self.encoder(src)
        return self.decoder(trg, enc=enc, mask=trg_mask)
    
    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int, temp: float = 0.3):

        # Get encoder embeddings and start tokens
        enc = self.encoder(src)
        import matplotlib.pyplot as plt
        plt.imshow(enc[0].detach().cpu().numpy())
        plt.colorbar()
        plt.savefig('enc.png')
        plt.close()

        start_tokens = torch.LongTensor([self.bos_token] * src.shape[0]).unsqueeze(1).to(self.device)

        # Generate using auto-regressive transformer decoder
        return self.decoder.generate(
            start_tokens=start_tokens,
            eos_tok=self.eos_token,
            max_len=max_len,
            temp=temp,
            enc=enc
        )
    

def create_model(config: dict) -> OCRModel:
    """Create an OCRModel from a configuration file."""
    encoder = create_encoder(config)
    decoder = create_decoder(config)
    device = torch.device(config['device'])
    bos_token = config['bos_token']
    eos_token = config['eos_token']

    model = OCRModel(
        encoder,
        decoder,
        bos_token=bos_token,
        eos_token=eos_token,
        trg_pad_idx=config['trg_pad_idx'],
        device=device
    )

    return model