import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time as time

from model import OCRModel
from utils import load_config, create_model, count_parameters, get_optimizer, get_loss_fn, save_checkpoint, load_checkpoint
from dataset import ImagePadding, ImageDataset, DatasetSplits, BatchCollator, BucketBatchSampler, load_datasets, create_dataloader
from torch.utils.data import DataLoader


def test_model(test_loader: DataLoader, model: OCRModel, device: torch.device, verbose: bool = True):
    """Test the TeXOCR model."""
    model.eval()

    test_loss = 0
    test_acc = 0
    n_test = 0

    with torch.no_grad():
        for i, (img, trg) in enumerate(test_loader):

            if i in [0, 1, 2]:
                print(trg)
                print(trg.shape)
                print(img.shape)

            # if i == 1:
            #     # Plot first image in the batch

            #     print(img[120])

            #     import matplotlib.pyplot as plt
            #     plt.imshow(img[120].permute(1, 2, 0))
            #     plt.show()

            if verbose:
                print(f'Batch {i+1}/{len(test_loader)}', end='\r')

            img = img.to(device)
            trg = trg.to(device)

            out = model(img, trg[:, :-1])

            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), trg[:, 1:].reshape(-1), ignore_index=999)
            test_loss += loss.item()

            acc = (out.argmax(2) == trg[:, 1:]).sum() / (trg[:, 1:] != 999).sum()
            test_acc += acc.item()

            n_test += 1

    test_loss /= n_test
    test_acc /= n_test

    if verbose:
        print(f'Test loss: {test_loss:.4f}')
        print(f'Test accuracy: {test_acc:.4f}')

    model.train()

    return test_loss, test_acc

def single_prediction(model: OCRModel, test_loader: DataLoader, device: torch.device):
    """Make a single prediction."""
    model.eval()

    with torch.no_grad():
        for i, (img, trg) in enumerate(test_loader):

            img = img.to(device)
            trg = trg.to(device)

            out = model(img, trg[:, :-1])

            out = out[0].unsqueeze(0)
    
    print(out.argmax(2))
    model.train()

    return out

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_default_dtype(torch.float32)
    # Load config file
    config = load_config('config.yml')
    
    # Load datasets
    train_set, val_set, test_set = load_datasets('data')

    test_set.max_seq_len = test_set.get_max_seq_len()

    print(test_set.tokenizer_path)

    print(sorted([len(enc_label) for enc_label in [test_set.tokenizer.encode(label) for label in test_set.labels]])[:128])

    print(test_set.max_seq_len)
    print(test_set.get_max_seq_len())

    
    # Add max_length and vocab_size to config
    config['max_length'] = 839
    config['vocab_size'] = test_set.tokenizer.vocab_size
    
    # Get train and val dataloaders
    test_loader = create_dataloader(test_set, config)

    
    device = torch.device(config['device'])
    model = create_model(config).to(device) # Create model
    optimizer = get_optimizer(model, config) # Get optimizer

    model, optimizer, epoch = load_checkpoint(model, optimizer, device, 'checkpoints/checkpoint_newimg.pth')

    # Print model.pos_embedding and class_token weights

    n_params = count_parameters(model)

    print(f'Using device: {device}')
    print(f'Model has {n_params} parameters.')

    test_model(test_loader, model, device)

