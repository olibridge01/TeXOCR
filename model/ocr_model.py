import torch
import torch.nn as nn
import torch.nn.functional as F

from TeXOCR.model.encoder import VisionEncoder
from TeXOCR.model.decoder import TransformerDecoder

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
        # enc_src = self.encoder(src, mask=None)
        enc_src = self.encoder(src)
        out = self.decoder(trg, enc_src, src_mask=None, trg_mask=trg_mask)

        # return self.decoder(trg, context=enc_src, mask=None)

        return out
    
    def generate(self, start_token, max_length, eos_token, src):
        """
        Args:
            start_token: Start token.
            max_length: Maximum length.
            eos_token: End of sequence token.
        """

        src_mask = self.make_src_mask(src)
        # enc_out = self.encoder(src, mask=None)
        print(src.shape)
        enc_out = self.encoder(src)

        print(f'enc_out shape: {enc_out.shape}')
        # for i in range(64, 128):
        #     print(enc_out[0,i,:])

        import matplotlib.pyplot as plt
        # Plot all embedding vectors for the first image
        plt.figure(figsize=(10, 10))
        plt.imshow(enc_out[0].detach().numpy())
        plt.colorbar()
        plt.savefig('enc_out2.png')
        plt.close()


        trg_tensor = torch.tensor([start_token] * src.shape[0]).unsqueeze(1).to(self.device)
        seen_eos = torch.tensor([False] * src.shape[0]).to(self.device)

        for i in range(max_length):
            
            # For generations, target mask is not needed. Generate tensor for target mask


            out = self.decoder(trg_tensor, enc_out, src_mask=None, trg_mask=None)

            out = out.argmax(2)[:, -1].unsqueeze(1)

            trg_tensor = torch.cat((trg_tensor, out), 1)

            seen_eos |= (out == eos_token).flatten()

            if seen_eos.all():
                break

        return trg_tensor
    
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

    