import argparse
import time as time
from typing import Dict, List, Tuple, Callable, Any

from TeXOCR.data_wrangling.dataset import ImageDataset
from TeXOCR.utils import load_config

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Dataset utilities.")
    parser.add_argument('-c', '--config', type=str, default='config/config.yml', help='Path to the configuration file.')
    parser.add_argument('--split', type=str, default='train', help='Split to create Dataset from.')
    parser.add_argument('-s', '--save', type=str, default='dataset.pkl', help='Path to save the dataset.')
    args = parser.parse_args()

    assert args.split in ['train', 'val', 'test'], "Split must be one of 'train', 'val', or 'test'."
    return args

def main(args: argparse.Namespace) -> None:
    """Extract the dataset and create the splits."""
    start = time.time()

    config = load_config(args.config) # Get configs

    if args.split == 'train':
        root_dir = config['train_dir']
    elif args.split == 'val':
        root_dir = config['val_dir']
    else:
        root_dir = config['test_dir']

    # Load the dataset
    dataset = ImageDataset(
        root_dir=root_dir,
        tokenizer_path=config['tokenizer_path'], 
        dataset_size=config['num_equations']
    )

    # Save dataset
    dataset.save(f"{args.save}")

    print(f"Dataset of size {len(dataset)} saved to {args.save}.")
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds.")

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)