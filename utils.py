import yaml
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from TeXOCR.model import VisionEncoder, TransformerDecoder, OCRModel
from TeXOCR.hybrid_encoder import CustomVisionTransformer
# from TeXOCR.x_decoder import CustomARWrapper

from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_hybrid import HybridEmbed
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from einops import repeat

from x_transformers import TransformerWrapper, Decoder

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

# def create_encoder(config: dict) -> VisionEncoder:
#     """Create a VisionEncoder from a config dict."""
#     batch_size = config['batch_size']
#     patch_size = config['patch_size']
#     encoder_args = config['encoder']
#     device = torch.device(config['device'])

#     encoder = VisionEncoder(
#         patch_size=patch_size,
#         batch_size=batch_size,
#         device=device,
#         **encoder_args
#     )

#     return encoder

def create_encoder(config: dict):
    """Create a VisionEncoder from a config dict."""
    batch_size = config['batch_size']
    patch_size = config['patch_size']
    encoder_args = config['encoder']
    device = torch.device(config['device'])

    backbone = ResNetV2(
        layers=[2,3,7], num_classes=0, global_pool='', in_chans=1,
        preact=False, stem_type='same', conv_layer=StdConv2dSame)
    min_patch_size = 2**(3+1)

    def embed_layer(**x):
        ps = x.pop('patch_size', min_patch_size)
        assert ps % min_patch_size == 0 and ps >= min_patch_size, 'patch_size needs to be multiple of %i with current backbone configuration' % min_patch_size
        return HybridEmbed(**x, patch_size=ps//min_patch_size, backbone=backbone)

    encoder = CustomVisionTransformer(img_size=(160, 1008),
                                      patch_size=patch_size,
                                      in_chans=1,
                                      num_classes=0,
                                      embed_dim=256,
                                      depth=4,
                                      num_heads=8,
                                      embed_layer=embed_layer
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

# def create_decoder(config: dict):
#     """Create a TransformerDecoder from a config dict."""
#     assert 'max_length' in config, 'max_length not loaded into config file!'
#     assert 'vocab_size' in config, 'vocab_size not loaded into config file!'

#     device = torch.device(config['device'])
#     max_length = config['max_length']
#     vocab_size = config['vocab_size']
#     decoder_args = {
#         'attn_on_attn': True,
#         'cross_attend': True,
#         'ff_glu': True,
#         'rel_pos_bias': False,
#         'use_scalenorm': False
#     }

#     return CustomARWrapper(
#         TransformerWrapper(
#             num_tokens=vocab_size,
#             max_seq_len=max_length,
#             attn_layers=Decoder(
#                 dim=256,
#                 depth=4,
#                 heads=8,
#                 **decoder_args
#             )),
#         pad_value=999)
    

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

def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, device: torch.device, load_path: str) -> Tuple[nn.Module, optim.Optimizer, int]:
    """Load a model checkpoint."""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print('Checkpoint loaded!')

    return model, optimizer, epoch


if __name__ == '__main__':
    config = load_config('config.yml')
    alphabetize_config(config)