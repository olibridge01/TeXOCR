import random
import argparse
from pathlib import Path
from typing import Tuple, List

from TeXOCR.utils import load_config

def split_data(input_file: str, splits: Tuple[float, float, float], output_dir: str, num_equations: int, seed: int = 42, verbose: bool = True):
    """
    Split the data into train, test, and val sets.

    Args:
        input_file: Path to the original .txt file containing LaTeX equations.
        splits: Tuple containing the ratios for train, test, and val sets.
        output_dir: Directory to save the split files.
        seed: Random seed for reproducibility.
    """
    label_name = 'labels.txt'

    # Ensure that the sum of the ratios is 1
    assert sum(splits) == 1, "The sum of the splits must be equal to 1."
    train_ratio, test_ratio, val_ratio = splits

    # Read the input file and strip of any leading/trailing whitespaces and newlines
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # Shuffle the data
    random.seed(seed)
    random.shuffle(lines)

    # Calculate the number of samples for each split
    total_lines = min(num_equations, len(lines))
    lines = lines[:total_lines]

    train_size = int(total_lines * train_ratio)
    test_size = int(total_lines * test_ratio)
    val_size = total_lines - train_size - test_size

    if verbose:
        print(f'Splitting data: {train_size} train | {test_size} test | {val_size} val')

    # Split the data
    train_data = lines[:train_size]
    test_data = lines[train_size:train_size + test_size]
    val_data = lines[train_size + test_size:]

    # Create output directories if they don't exist
    output_dir = Path(output_dir)
    output_dir.joinpath('train').mkdir(parents=True, exist_ok=True)
    output_dir.joinpath('test').mkdir(parents=True, exist_ok=True)
    output_dir.joinpath('val').mkdir(parents=True, exist_ok=True)

    # Write the data into corresponding files
    write_data(output_dir.joinpath('train', label_name), train_data)
    write_data(output_dir.joinpath('test', label_name), test_data)
    write_data(output_dir.joinpath('val', label_name), val_data)

def write_data(output_file: str, data: List[str]):
    """Output equation data to a file"""
    with open(output_file, 'w') as f:
        for line in data:
            f.write(line + '\n')

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Shuffle and split LaTeX equations into train, test, and val sets.")
    parser.add_argument('input_file', type=str, help="Path to the .txt file containing LaTeX equations.")
    parser.add_argument('output_dir', type=str, help="Directory to save the split files.")
    parser.add_argument('-c', '--config', type=str, default='config/data_config.yml', help="Path to the configuration file.")
    return parser.parse_args()

def main(args: argparse.Namespace):
    """Main function to split LaTeX equations into train, test and val sets."""
    config = load_config(args.config)

    input_file = args.input_file
    output_dir = args.output_dir
    splits = config['splits']
    num_equations = config['num_equations']
    seed = config['seed']

    splits = tuple([float(split) for split in splits.values()])
    split_data(input_file, splits, output_dir, num_equations, seed)

if __name__ == "__main__":
    args = parse_args()
    main(args)