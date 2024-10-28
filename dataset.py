import os
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler

from torchvision import transforms

from tokenizer import BaseTokenizer, RegExTokenizer
from PIL import Image


class BatchCollator(object):
    """Batch collator for the DataLoader. Pads each batch to the maximum sequence length."""
    def __init__(self, pad_token: int):
        """
        Args:
            pad_token: Padding token id.
        """
        self.pad_token = pad_token

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: List of tuples containing images and labels, sampled from the Dataset.
        """
        images, seq = zip(*batch)
        images = torch.stack(images)
        
        # Pad labels
        max_len = max([s.shape[0] for s in seq])
        padded_labels = torch.zeros(len(seq), max_len, dtype=torch.long) + self.pad_token

        for i, s in enumerate(seq):
            padded_labels[i, :s.shape[0]] = s

        return images, padded_labels


class ImagePadding(object):
    """Pad images to be a certain size."""
    def __init__(self, height: int, width: int, fill: int = 255):
        """
        Args:
            height: Desired image height.
            width: Desired image width.
            fill: Fill value for the padding.
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
    

class ImageDataset(Dataset):
    """Custom dataset class for image data and their corresponding tokenized LaTeX strings."""

    pad_char = '<PAD>'
    fixed_height = 160

    def __init__(self, root_dir: str, tokenizer_path: str, dataset_size: int, patch_size: int = 16, batch_size: int = 32):
        """
        Args:
            root_dir: Path to the root directory containing the image data.
            tokenizer_path: Path to the tokenizer file.
            dataset_size: Number of samples in the dataset.
            patch_size: Size of the patches to be used in the ViT encoder.
            batch_size: Size of the batch.
        """
        # Initial variables
        self.dataset_size = dataset_size
        
        # Load tokenizer
        self.tokenizer = RegExTokenizer()
        self.tokenizer.load(tokenizer_path)

        # Sort dataset paths
        self.root_dir = Path(root_dir) # Root directory as Path object
        self.images_path = self.root_dir / 'images'
        self.label_path = self.root_dir / 'labels.txt'
        self.id_path = self.root_dir / 'ids.txt'

        # Load image ids and labels from respective files
        self.image_ids = self._load_ids()
        self.labels = self._load_labels()

        # Get maximum height, width and sequence length
        self.max_height, self.max_width = self.get_max_dims()
        self.max_seq_len = self.get_max_seq_len()

        print(f"Max height: {self.max_height}, Max width: {self.max_width}")
        print(f"Max sequence length: {self.max_seq_len}")

        # Scaled height and width
        scaled_height = self.fixed_height
        scaled_width = int(self.max_width * (self.fixed_height / self.max_height))

        # Transfomations consist of converting to Tensor, converting to grayscale, and adding padding to make all images the same size
        h, w = self.nearest_patch_multiple(patch_size, scaled_height, scaled_width)
        print(f"Resized image dimensions: {h}, {w}")
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=h, max_size=w),
            ImagePadding(h, w)
        ])

        # Label padding 
        pad_token = self.tokenizer.special_tokens[self.pad_char]
        # self.label_transform = SequencePadding(self.max_seq_len, pad_value=pad_token)

    def nearest_patch_multiple(self, patch_size: int, height: int, width: int) -> Tuple[int, int]:
        """Calculate the nearest multiple of the patch size for the height and width of the image. Round up."""
        h = patch_size * ((height + patch_size - 1) // patch_size)
        w = patch_size * ((width + patch_size - 1) // patch_size)

        return h, w

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
        return max([len(self.tokenizer.encode(label)) for label in self.labels])

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
            # This may be a useful place for a warning or logging message
        
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
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.dataset_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a sample from the dataset."""

        # Check if the index is a tensor
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image path
        image_path = self.images_path / self.image_ids[idx]

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Apply transform if it exists
        image = self.img_transform(image)

        label_str = self.labels[idx] # Get LaTeX string
        label_enc = torch.tensor(self.tokenizer.encode(label_str), dtype=torch.long) # Encode LaTeX string to tokens

        # Apply label transform
        # label_enc = self.label_transform(label_enc)

        return image, label_enc
    

class BucketBatchSampler(BatchSampler):
    """Batch sampler that batches samples of similar sequence length together."""
    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool = False):
        """
        Initialize the sampler.
        
        Args:
            dataset: Dataset to sample from.
            batch_size: Size of the batch.
            drop_last: Whether to drop the last batch if it is smaller than batch_size.
        """
        super(BucketBatchSampler, self).__init__(dataset, batch_size, drop_last)

        # Sort the dataset by sequence length
        self.sorted_indices = self._sort_indices(dataset)

        self.batch_size = batch_size
        self.drop_last = drop_last

    def _sort_indices(self, dataset: Dataset) -> List[int]:
        """Sort the indices of the dataset by sequence length."""
        indices = list(range(len(dataset)))
        indices.sort(key=lambda x: [len(dataset.tokenizer.encode(dataset.labels[x]))])

        return indices

    def __iter__(self):
        batch = []
        for idx in self.sorted_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sorted_indices) // self.batch_size
        else:
            return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size
        
    
if __name__ == '__main__':

    # Test the ImageDataset class
    dataset = ImageDataset(root_dir='data', tokenizer_path='tokenizer.txt', dataset_size=1000)
    
    # Test the BucketBatchSampler and BatchCollator classes
    batch_sampler = BucketBatchSampler(dataset, batch_size=32)
    collate_fn = BatchCollator(dataset.tokenizer.special_tokens[dataset.pad_char])
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    for i, (images, labels) in enumerate(dataloader):
        if i == 20:
            print(images.shape)
            print(labels.shape)

            # Plot images on a grid
            
            import matplotlib.pyplot as plt
            from torchvision.utils import make_grid

            grid = make_grid(images, nrow=8)
            # plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0))

            # Plot a 16*16 grid over the image
            h, w = images.shape[2], images.shape[3]
            for i in range(15, h*4, 16):
                plt.axhline(i, color='r', linewidth=0.3)
            for i in range(15, w*8, 16):
                plt.axvline(i, color='r', linewidth=0.3)

            # Scale up axes by 2
            ax = plt.gca()
            ax.set_xlim([0, w*8])
            ax.set_ylim([h*4, 0])

            plt.show()

            break