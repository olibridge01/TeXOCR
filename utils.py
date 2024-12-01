import re
import yaml
import math
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from einops import repeat

def exists(x):
    """Check for existence of x."""
    return x is not None

def get(x, y):
    """x if it exists, otherwise y."""
    if exists(x):
        return x
    return y

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
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print('Checkpoint loaded!')

    return model, optimizer, epoch

def process_output(output: str) -> str:
    """Postprocessing for the LaTeX output to make it more human-readable."""
    output = re.sub(r'(\\[a-zA-Z]+)\s+([a-zA-Z0-9])', r'\1<SPACE>\2', output)
    output = re.sub(r'\s+', '', output)
    output = output.replace('<SPACE>', ' ')

    return output

def max_negative_val(x):
    """Maximum negative value for a tensor."""
    return -torch.finfo(x.dtype).max

def topk(logits: torch.Tensor, threshold: float = 0.9):
    """Top-k filtering for logits."""
    k = int((1 - threshold) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **kwargs) -> int:
    """Get padding for a convolutional layer."""
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def get_same_padding(x: int, k: int, s: int, d: int) -> int:
    """Get padding for a convolutional layer to maintain spatial dimensions."""
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **kwargs) -> bool:
    """Check if padding is static."""
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

def padding_val(padding: str, kernel_size: int, **kwargs) -> Tuple[int, bool]:
    dynamic = False
    """Get padding value and whether it is dynamic."""
    if is_static_pad(kernel_size, **kwargs):
        padding = get_padding(kernel_size, **kwargs)
    else:
        padding = 0
        dynamic = True
    return padding, dynamic

def pad_same(x: torch.Tensor, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0.) -> torch.Tensor:
    """Pad a tensor to maintain spatial dimensions."""
    ih, iw = x.size()[-2:]
    pad_h = get_same_padding(ih, k[0], s[0], d[0])
    pad_w = get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


if __name__ == '__main__':
    config = load_config('config.yml')
    alphabetize_config(config)