import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# from TeXOCR.model.encoder import create_encoder
from TeXOCR.model import create_encoder, create_decoder, PositionalEmbedding
from TeXOCR.tokenizer import RegExTokenizer
from TeXOCR.utils import load_config, process_output
from TeXOCR.data_wrangling.dataset import img_transform


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

        # Get encoder embeddings and start tokens
        enc = self.encoder(src)
        # import matplotlib.pyplot as plt
        # plt.imshow(enc[0].detach().cpu().numpy())
        # plt.colorbar()
        # plt.savefig('enc.png')
        # plt.close()

        start_tokens = torch.LongTensor([self.bos_token] * src.shape[0]).unsqueeze(1).to(self.device)

        # Generate using auto-regressive transformer decoder
        return self.decoder.generate(
            start_tokens=start_tokens,
            eos_tok=self.eos_token,
            max_len=max_len,
            temp=temp,
            enc=enc
        )
    

class TeXOCRWrapper(object):
    """TeXOCR model wrapper for command-line inference and use in the web app."""
    def __init__(self, config: dict):

        # Load tokenizer
        self.tokenizer = RegExTokenizer()
        self.tokenizer.load(config['tokenizer_path'])

        # Add vocab size to config
        config['vocab_size'] = self.tokenizer.vocab_size

        # Create model
        self.model = create_model(config)
        model_state_dict = torch.load(config['model_path'], map_location=config['device'], weights_only=True)

        if 'decoder.net.pos_embedding.embedding.weight' in model_state_dict:
            pos_embed_len, embed_dim = model_state_dict['decoder.net.pos_embedding.embedding.weight'].shape

            # Re-adjust model to match the pos-embedding length
            self.model.decoder.net.pos_embedding = PositionalEmbedding(embed_dim, pos_embed_len)

        self.model.load_state_dict(model_state_dict)

        self.img_transform = img_transform

    def __call__(self, img: Image.Image, max_len: int = 350, temp: float = 0.3) -> str:
        """Convert an image to LaTeX."""
        # Convert to tensor and add batch dimension
        img = self.img_transform(img)
        img = img.unsqueeze(0).to(self.model.device)

        # Generate LaTeX prediction
        pred_tokens = self.model.generate(img, max_len=max_len, temp=temp)
        
        # Convert to list and strip the <EOS> token
        out_tokens = pred_tokens.squeeze(0).tolist()[:-1]
        out_str = self.tokenizer.decode(out_tokens)

        # Post-process the output to remove appropriate whitespace
        out_str = process_output(out_str)

        return out_tokens, out_str


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

if __name__ == '__main__':

    config = load_config('config/final_config.yml')
    model = TeXOCRWrapper(config)

    img = Image.open('data/val/images/eq_000833.png')

    model(img)