import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torcheval.metrics.functional import bleu_score
from torchvision import transforms

import time as time
from tqdm import tqdm

from TeXOCR.model import OCRModel
from TeXOCR.utils import load_config, create_model, count_parameters, get_optimizer, get_loss_fn, save_checkpoint, load_checkpoint
from TeXOCR.dataset import ImagePadding, ImageDataset, BatchCollator, BucketBatchSampler, Invert, load_datasets, create_dataloader


def test_model(test_loader: DataLoader, model: OCRModel, device: torch.device, verbose: bool = True):
    """Test the TeXOCR model."""
    model.eval()

    test_loss = 0
    test_acc = 0
    n_test = 0

    import matplotlib.pyplot as plt

    with torch.no_grad():
        for i, (img, trg) in enumerate(tqdm(test_loader, desc='Testing')):

            # if i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
            #     print(trg)
            #     print(trg.shape)
            #     print(img.shape)

            #     # Plot first image in the batch
            #     plt.imshow(img[10].permute(1, 2, 0), cmap='gray')
            #     plt.colorbar()

            #     # Plot 16*16 grid of patches on top of the image
            #     for j in range(0, 160, 16):
            #         for k in range(0, 1008, 16):
            #             plt.plot([k, k+16], [j, j], color='r', linewidth=0.5)
            #             plt.plot([k, k+16], [j+16, j+16], color='r', linewidth=0.5)
            #             plt.plot([k, k], [j, j+16], color='r', linewidth=0.5)
            #             plt.plot([k+16, k+16], [j, j+16], color='r', linewidth=0.5)

            #     plt.savefig(f'img_{i}.png')
            #     plt.close()

            # if i == 21:

            #     img = img.to(device)
            #     trg = trg.to(device)
            #     print(trg.shape)
            #     pred = model.generate(start_token=998, max_length=270, eos_token=997, src=img)

            #     for j in [92, 104]:

            #         target = trg[j]
            #         prediction = pred[j]

            #         eos_index = (target==997).nonzero().squeeze().int().item()

            #         corr = (target[:eos_index+1] == prediction[:eos_index+1]).sum() / (eos_index+1)
            #         print(f'{j}, Acc: {corr:.3f}')

            #         print(prediction[:eos_index+1])
            #         print(target)

            #         print(test_set.tokenizer.decode(prediction[:eos_index+1].tolist()))
            #         print(test_set.tokenizer.decode(target.tolist()))

            #     exit()

            # if verbose:
            #     print(f'Batch {i+1}/{len(test_loader)}', end='\r')

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

    print(len(test_set))
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

    model, optimizer, epoch = load_checkpoint(model, optimizer, device, 'checkpoints/checkpoint_hybrid.pth')

    # Print model.pos_embedding and class_token weights

    n_params = count_parameters(model)

    print(f'Using device: {device}')
    print(f'Model has {n_params} parameters.')

    test_model(test_loader, model, device, test_set)

