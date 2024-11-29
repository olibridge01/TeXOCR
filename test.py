import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torcheval.metrics.functional import bleu_score
from torchvision import transforms

import time as time
from tqdm import tqdm

from TeXOCR.model import OCRModel, create_modelTeXOCR
from TeXOCR.utils import load_config, count_parameters, get_optimizer, get_loss_fn, save_checkpoint, load_checkpoint
from TeXOCR.data_wrangling.dataset import ImagePadding, ImageDataset, BatchCollator, BucketBatchSampler, Invert, load_datasets, create_dataloader
from TeXOCR.eval.eval import batch_acc


def test_model(test_loader: DataLoader, model: OCRModel, device: torch.device, test_set, verbose: bool = True):
    """Test the TeXOCR model."""
    model.eval()

    # test_loss = 0
    test_acc = 0
    n_test = 0

    # import matplotlib.pyplot as plt

    with torch.no_grad():
        for i, (img, trg) in enumerate(tqdm(test_loader, desc='Testing')):

            img = img.to(device)
            trg = trg.to(device)
            print(trg.shape)
            pred = model.generate(max_len=276, src=img)

            acc = batch_acc(pred, trg, pad_token=999)
            print(f'Batch accuracy: {acc:.3f}')
            test_acc += acc

            for j in range(1):

                target = trg[j]
                prediction = pred[j]

                eos_index = (target==997).nonzero().squeeze().int().item()

                print(eos_index)
                print(prediction.shape)
                print(target.shape)

                # corr = (target[:eos_index+1] == prediction[:eos_index+1]).sum() / (eos_index+1)
                # print(f'{j}, Acc: {corr:.3f}')

                print(prediction[:eos_index+1])
                print(target)

                print(test_set.tokenizer.decode(prediction[:eos_index+1].tolist()))
                print(test_set.tokenizer.decode(target.tolist()))

            n_test += 1

    # test_loss /= n_test
    test_acc /= n_test

    if verbose:
        # print(f'Test loss: {test_loss:.4f}')
        print(f'Test accuracy: {test_acc:.4f}')

    model.train()

    return test_acc

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
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.set_default_dtype(torch.float32)
    # Load config file
    config = load_config('config/config.yml')
    
    # Load datasets
    train_set, val_set, test_set = load_datasets('data')

    test_set.max_seq_len = test_set.get_max_seq_len()

    print(test_set.tokenizer_path)

    # print(sorted([len(enc_label) for enc_label in [test_set.tokenizer.encode(label) for label in test_set.labels]])[:128])

    print(len(test_set))
    print(test_set.max_seq_len)
    print(test_set.get_max_seq_len())

    
    # Add max_length and vocab_size to config
    config['max_length'] = 859
    config['vocab_size'] = test_set.tokenizer.vocab_size
    
    # Get train and val dataloaders
    test_loader = create_dataloader(test_set, config)

    
    device = torch.device(config['device'])
    model = create_model(config).to(device) # Create model
    optimizer = get_optimizer(model, config) # Get optimizer

    model, optimizer, epoch = load_checkpoint(model, optimizer, device, 'checkpoints/checkpoint_eureka.pth')

    # Print model.pos_embedding and class_token weights

    n_params = count_parameters(model)

    print(f'Using device: {device}')
    print(f'Model has {n_params} parameters.')

    test_model(test_loader, model, device, test_set)

