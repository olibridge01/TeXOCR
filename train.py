import argparse
import time as time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from TeXOCR.model import OCRModel, create_model
from TeXOCR.test import test_model
from TeXOCR.utils import load_config, count_parameters, get_optimizer, get_loss_fn, save_checkpoint
from TeXOCR.data_wrangling.dataset import ImagePadding, Invert, ImageDataset, BatchCollator, BucketBatchSampler, load_datasets, create_dataloader

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TeXOCR model.')
    parser.add_argument('-d', '--data_dir', type=str, default='data', help='Path to the directory containing dataset pickle files.')
    parser.add_argument('--config', type=str, default='config/config.yml', help='Path to the configuration file.')
    return parser.parse_args()

def main(args: argparse.Namespace):
    """Main training function."""
    
    config = load_config(args.config) # Get configs

    print('Loading datasets...')
    train_set, val_set, test_set = load_datasets(args.data_dir) # Load datasets from pickle files
    print('Datasets loaded!')

    # Add max_length and vocab_size to config
    config['max_length'] = train_set.max_seq_len
    config['vocab_size'] = train_set.tokenizer.vocab_size

    # Get train and val dataloaders
    train_loader = create_dataloader(train_set, config)
    val_loader = create_dataloader(val_set, config)

    train_model(train_loader, val_loader, config)

def train_model(train_loader: DataLoader, val_loader: DataLoader, config: dict, verbose: bool = True):
    """Train the TeXOCR model."""

    device = torch.device(config['device'])
    model = create_model(config).to(device) # Create model
    n_params = count_parameters(model)

    if verbose:
        print(f'Using device: {device}')
        print(f'Model has {n_params} parameters.')

    optimizer = get_optimizer(model, config) # Get optimizer
    criterion = get_loss_fn(config)(ignore_index=config['trg_pad_idx']) # Get loss function

    n_epochs = config['n_epochs']
    if verbose:
         print('Training model...')

    start = time.time()
    for epoch in range(n_epochs):
        
        model.train()
        epoch_loss = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')):

            # if verbose:
            #     print(f'Batch {i+1}/{len(train_loader)}')
            
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)

            # print(f'Images shape: {images.shape}')
            
            optimizer.zero_grad()
            
            loss = model(images, targets)
            
            loss.backward()
            optimizer.step()

            # print(f'Loss: {loss.item()}')
            
            epoch_loss += loss.item()
        
        if verbose:
            print(f'Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss / len(train_loader)}')

        if config['save_checkpoint'] and (epoch+1) % config['save_freq'] == 0:
            save_checkpoint(model, optimizer, epoch, 'checkpoints')

        if (epoch+1) % config['val_freq'] == 0:
            test_model(val_loader, model, device, verbose=True)

    end = time.time()
    if verbose:
        print(f'Training took {(end - start):.2f} seconds.')


if __name__ == '__main__':
    args = parse_args()
    main(args)