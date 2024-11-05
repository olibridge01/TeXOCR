import yaml
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import VisionEncoder, TransformerDecoder, OCRModel

def load_config(config_path: str) -> dict:
    """Load a configuration file in yaml format."""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def alphabetize_config(config: dict) -> dict:
    """Alphabetize a configuration file and resave it."""
    config = dict(sorted(config.items()))
    with open('config.yml', 'w') as f:
        yaml.dump(config, f)
    return config

def create_encoder(config: dict) -> VisionEncoder:
    """Create a VisionEncoder from a config dict."""
    batch_size = config['batch_size']
    patch_size = config['patch_size']
    encoder_args = config['encoder']
    device = torch.device(config['device'])

    encoder = VisionEncoder(
        patch_size=patch_size,
        batch_size=batch_size,
        device=device,
        **encoder_args
    )

    return encoder

def create_decoder(config: dict) -> TransformerDecoder:
    """Create a TransformerDecoder from a config dict."""
    assert 'max_length' in config, 'max_length not loaded into config file!'
    assert 'vocab_size' in config, 'vocab_size not loaded into config file!'

    device = torch.device(config['device'])
    max_length = config['max_length']
    vocab_size = config['vocab_size']
    decoder_args = config['decoder']

    decoder = TransformerDecoder(
        device=device,
        max_length=max_length,
        vocab_size=vocab_size,
        **decoder_args
    )

    return decoder

def create_model(config: dict) -> OCRModel:
    """Create an OCRModel from a configuration file."""
    encoder = create_encoder(config)
    decoder = create_decoder(config)
    device = torch.device(config['device'])

    model = OCRModel(
        encoder,
        decoder,
        src_pad_idx=config['src_pad_idx'],
        trg_pad_idx=config['trg_pad_idx'],
        device=device
    )

    return model

def count_parameters(model: nn.Module) -> int:
    """Count the number of parameters in a model."""
    return sum([p.numel() for p in model.parameters()])

def get_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Get an optimizer from a configuration file."""
    optimizer_class = getattr(optim, config['optimizer'])
    optimizer_args = config['optimizer_args']

    return optimizer_class(model.parameters(), **optimizer_args)

def get_loss_fn(config: dict) -> nn.Module:
    """Get a loss function from a configuration file."""
    return getattr(nn, config['loss_fn'])

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, save_dir: str):
    """Save a model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }

    save_path = f'{save_dir}/checkpoint_e{epoch}.pth'
    torch.save(checkpoint, save_path)


if __name__ == '__main__':
    config = load_config('config.yml')
    alphabetize_config(config)