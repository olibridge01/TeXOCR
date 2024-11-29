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

    


        # trg_tensor = torch.tensor([start_token] * src.shape[0]).unsqueeze(1).to(self.device)
        # seen_eos = torch.tensor([False] * src.shape[0]).to(self.device)

        # for i in range(max_length):
            
        #     # For generations, target mask is not needed. Generate tensor for target mask


        #     out = self.decoder(trg_tensor, enc_out, src_mask=None, trg_mask=None)

        #     out = out.argmax(2)[:, -1].unsqueeze(1)

        #     trg_tensor = torch.cat((trg_tensor, out), 1)

        #     seen_eos |= (out == eos_token).flatten()

        #     if seen_eos.all():
        #         break

        # return trg_tensor

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

    