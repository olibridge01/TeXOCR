import time as time
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from typing import Dict, List, Tuple, Callable, Any
import os, pickle, random, argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler

from torchvision import transforms

from TeXOCR.tokenizer import RegExTokenizer
from TeXOCR.utils import load_config


class BatchCollator(object):
    """Batch collator for the DataLoader. Pads each batch to the maximum sequence length."""
    def __init__(self, pad_token: int, bos_token: int, eos_token: int, shuffle: bool = False, seed: int = 42):
        """
        Args:
            pad_token: Padding token id.
        """
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.shuffle = shuffle
        self.starting_seed = seed
        self.seed = seed

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: List of tuples containing images and labels, sampled from the Dataset.
        """
        images, seq = zip(*batch)

        if self.shuffle:
            random.seed(self.seed)
            indices = list(range(len(images)))
            random.shuffle(indices)
            self.seed += 1

            images = [images[i] for i in indices]
            seq = [seq[i] for i in indices]

        images = torch.stack(images)

        # Pad labels
        max_len = max([s.shape[0] for s in seq]) + 2 # Add 2 for <BOS> and <EOS> tokens
        padded_labels = torch.zeros(len(seq), max_len, dtype=torch.long) + self.pad_token

        for i, s in enumerate(seq):
            padded_labels[i, 0] = self.bos_token
            padded_labels[i, 1:s.shape[0]+1] = s
            padded_labels[i, s.shape[0]+1] = self.eos_token

        return images, padded_labels
    

class Invert(object):
    """Invert the colors of an image."""
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Image tensor to be inverted.
        """
        return 1 - img
    
    def __repr__(self) -> str:
        return "Invert()"


class ImagePadding(object):
    """Pad images to be a certain size."""
    def __init__(self, height: int, width: int, fill: int = 0):
        """
        Args:
            height: Desired image height.
            width: Desired image width.
            fill: Fill value for the padding. Default is 0.
        """
        self.height = height
        self.width = width
        self.fill = fill

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Image tensor to be padded.
        """
        pad_h = self.height - img.shape[1]
        pad_w = self.width - img.shape[2]

        # Centred padding
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        img = F.pad(img, padding, value=self.fill)

        return img
    
    def __repr__(self) -> str:
        return f"ImagePadding(height={self.height}, width={self.width}, fill={self.fill})"
    

class ImageDataset(Dataset):
    """Custom dataset class for image data and their corresponding tokenized LaTeX strings."""

    pad_char = '<PAD>'
    bos_char = '<BOS>'
    eos_char = '<EOS>'
    # fixed_height = 160

    def __init__(self, root_dir: str = None, tokenizer_path: str = None, dataset_size: int = None, patch_size: int = 16):
        """
        Args:
            root_dir: Path to the root directory containing the image data.
            tokenizer_path: Path to the tokenizer file.
            dataset_size: Number of samples in the dataset.
            patch_size: Size of the patches to be used in the ViT encoder.
            batch_size: Size of the batch.
        """

        if all([root_dir, tokenizer_path, dataset_size]):
            # Load tokenizer
            self.tokenizer_path = tokenizer_path
            self.tokenizer = RegExTokenizer()
            self.tokenizer.load(tokenizer_path)

            # Sort dataset paths
            self.root_dir = Path(root_dir) # Root directory as Path object
            self.images_path = self.root_dir / 'images'

            # Check if labels and ids have been pruned during dataset generation
            if (self.root_dir / 'labels_pruned.txt').exists():
                self.label_path = self.root_dir / 'labels_pruned.txt'
                self.id_path = self.root_dir / 'ids_pruned.txt'
            else:
                self.label_path = self.root_dir / 'labels.txt'
                self.id_path = self.root_dir / 'ids.txt'

            # Allow truncation of data provided in root_dir
            num_lines = self._get_dataset_size()
            self.dataset_size = min(num_lines, dataset_size)

            # Load image ids and labels from respective files
            self.image_ids = self._load_ids()
            self.labels = self._load_labels()
            self.images = self.load_images()

            # Get maximum height, width and sequence length
            self.max_height, self.max_width = self.get_max_dims()
            self.max_seq_len = self.get_max_seq_len()

            print(f"Max height: {self.max_height}, Max width: {self.max_width}")
            print(f"Max sequence length: {self.max_seq_len}")

            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                Invert(),
            ])

    def _get_dataset_size(self) -> int:
        """Get the size of the dataset from the number of lines in the labels file."""
        with open(self.label_path, 'r') as f:
            return len(f.read().splitlines())

    def get_max_dims(self) -> Tuple[int, int]:
        """Get the maximum height and width of the images in the dataset."""
        max_height = 0
        max_width = 0

        for image_id in self.image_ids:
            image_path = self.images_path / image_id
            image = Image.open(image_path)
            width, height = image.size

            if height > max_height:
                max_height = height
            if width > max_width:
                max_width = width

        return max_height, max_width
    
    def get_max_seq_len(self) -> int:
        """Get the maximum sequence length of the tokenized labels in the dataset."""
        # Add 2 for the start and end tokens, <BOS> and <EOS>
        return max([len(self.tokenizer.encode(label)) for label in self.labels]) + 2

    def _load_ids(self) -> List[str]:
        """Load image ids from the ids file."""
        
        # Only retrieve the first dataset_size ids
        try:
            with open(self.id_path, 'r') as f:
                ids = f.read().splitlines()[:self.dataset_size]
        except IndexError:
            with open(self.id_path, 'r') as f:
                ids = f.read().splitlines()
            print(f"Dataset size is larger than the number of images in the dataset directory. Only {len(ids)} images will be used.")
        
        return ids
    
    def _load_labels(self) -> List[str]:
        """Load LaTeX strings from the labels file."""
        try:
            with open(self.label_path, 'r') as f:
                labels = f.read().splitlines()[:self.dataset_size]
        except IndexError:
            with open(self.label_path, 'r') as f:
                labels = f.read().splitlines()
            print(f"Dataset size is larger than the number of labels in the dataset directory. Only {len(labels)} labels will be used.")
        
        return labels
    
    def load_images(self) -> List:
        """Load images from the images directory."""
        images = []
        self.sizes = defaultdict(list)

        for i, image_id in enumerate(tqdm(self.image_ids, desc="Loading images")):

            image_path = self.images_path / image_id
            
            # Add copy of image to list
            image = Image.open(image_path)
            images.append(deepcopy(image))

            # Get image size and add to sizes dictionary
            w, h = image.size
            self.sizes[(w, h)].append(i)

            # Close the image to free up memory
            image.close()

        return images
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.dataset_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a sample from the dataset."""

        # Check if the index is a tensor
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the image
        image = self.images[idx]

        # Apply transform if it exists
        image = self.img_transform(image)

        label_str = self.labels[idx] # Get LaTeX string
        label_enc = torch.tensor(self.tokenizer.encode(label_str), dtype=torch.long) # Encode LaTeX string to tokens

        return image, label_enc
    
    def save(self, path: str):
        """Save the dataset to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str):
        """Load the dataset from a file."""
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        
        return dataset
    
    def __str__(self) -> str:
        """Return a string representation of the dataset."""
        return f"ImageDataset with {len(self)} samples."
    
    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"ImageDataset with {len(self)} samples."
    

class BucketBatchSampler(BatchSampler):
    """Batch sampler that batches samples of similar sequence length together."""
    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool = False, shuffle: bool = False, keep_small: bool = False, seed: int = 42):
        """
        Initialize the sampler.
        
        Args:
            dataset: Dataset to sample from.
            batch_size: Size of the batch.
            drop_last: Whether to drop the last batch if it is smaller than batch_size.
            shuffle: Whether to shuffle the batches.
        """
        super(BucketBatchSampler, self).__init__(dataset, batch_size, drop_last)
        self.sizes = dataset.sizes

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.keep_small = keep_small
        self.shuffle = shuffle
        self.starting_seed = seed
        self.seed = seed

    def __iter__(self):
        batches = []
        batch = []
        for size, ids in self.sizes.items():
            for i in range(0, len(ids), self.batch_size):
                batch = ids[i:i+self.batch_size]
                if len(batch) == self.batch_size or self.keep_small:
                    batches.append(batch)

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(batches)
            self.seed += 1

        for batch in batches:
            yield batch

    def __len__(self):
        if self.keep_small:
            length = sum([len(ids) // self.batch_size for ids in self.sizes.values()])
            length += sum([1 for ids in self.sizes.values() if len(ids) % self.batch_size > 0])
            return length
        else:
            return sum([len(ids) // self.batch_size for ids in self.sizes.values()])


def load_datasets(data_dir: str) -> Tuple[ImageDataset, ImageDataset, ImageDataset]:
    """Load train, validation, and test datasets from a given directory."""
    train_set = ImageDataset().load(f'{data_dir}/train/trainset.pkl')
    val_set = ImageDataset().load(f'{data_dir}/val/valset.pkl')
    # train_set = None
    # val_set = None
    test_set = ImageDataset().load(f'{data_dir}/test/testset.pkl')
    return train_set, val_set, test_set


def create_dataloader(dataset: ImageDataset, config: dict) -> torch.utils.data.DataLoader:
    """Create a DataLoader from a dataset and a configuration file."""
    pad, bos, eos = dataset.tokenizer.special_tokens.values()

    # Custom collate function for dataloader
    id_shuffle = config['id_shuffle']
    collate_fn = BatchCollator(pad, bos, eos, shuffle=id_shuffle)

    # Bucket batch sampler groups sequences of similar lengths together
    batch_shuffle = config['batch_shuffle']
    batch_size = config['batch_size']
    drop_last = config['drop_last']
    keep_small = config['keep_small']
    seed = config['seed']

    sampler = BucketBatchSampler(
        dataset, 
        batch_size=batch_size, 
        drop_last=drop_last, 
        shuffle=batch_shuffle, 
        keep_small=keep_small, 
        seed=seed
    )
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    
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
        dataset_size=config['dataset_size'], 
        patch_size=config['patch_size']
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