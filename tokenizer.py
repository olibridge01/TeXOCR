import argparse
import regex as re
from typing import List, Dict, Tuple

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BaseTokenizer:
    """Tokenizer base class."""
    def __init__(self, vocab_size: int = 800):
        
        self.vocab_size = vocab_size
        self.special_tokens = {}
        self.bp_merges = {}
        self.vocab = self._get_vocab()

    def _get_vocab(self) -> Dict[int, bytes]:
        """Given the current set of byte-pair merges, generate the Tokenizer's vocabulary."""
        vocab = {i: bytes([i]) for i in range(256)}

        # Add vocab for merged tokens
        for (i, j), token_id in self.bp_merges.items():
            vocab[token_id] = vocab[i] + vocab[j]
        
        # Add vocab for special tokens
        for token, token_id in self.special_tokens.items():
            vocab[token_id] = token.encode('utf-8')

        return vocab
    
    def _get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        """Given a list of token IDs, return a dictionary of token pairs and their counts."""
        stats = {}
        for i in range(1, len(ids)):
            sp = self.special_tokens.values()
            if ids[i-1] not in sp and ids[i] not in sp:
                stats[(ids[i-1], ids[i])] = stats.get((ids[i-1], ids[i]), 0) + 1
        return stats
    
    def _merge_tokens(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Merge a pair of tokens in a list of token IDs."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def encode(self, text: str) -> List[int]:
        """Produce a list of tokens from a text string."""
        
        # Tokenize text using the vocab and a utf-8 encoding
        ids = [id for id in text.encode('utf-8')]

        while True:
            
            if len(ids) < 2:
                break

            stats = self._get_stats(ids) # Get current counts of token pairs
            merge_pair = min(stats, key=lambda x: self.bp_merges.get(x, float('inf'))) # Get the pair with the lowest token id
            
            if merge_pair not in self.bp_merges:
                break
            
            # Conduct the merge
            new_id = self.bp_merges[merge_pair]
            ids = self._merge_tokens(ids, merge_pair, new_id)
        
        return ids

    def decode(self, tokens: List[int]) -> str:
        """Decode list of tokens to a text string."""
        return b''.join([self.vocab[token] for token in tokens]).decode('utf-8')
    
    def decode_list(self, tokens: List[int]) -> List[str]:
        """Decode list of tokens to a list of strings."""
        return [self.vocab[token].decode('utf-8') for token in tokens]
    
    def train(self, text: str, verbose: bool = False) -> None:
        """Train tokenizer."""
        UTF8_BYTE = 256
        
        # Tokenize text using the vocab and a utf-8 encoding
        ids = [id for id in text.encode('utf-8')]
        n_merges = self.vocab_size - UTF8_BYTE - len(self.special_tokens)
        merges = {}
        
        # Iterate through the token pairs and merge them
        for i in range(n_merges):
            stats = self._get_stats(ids)

            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            new_id = UTF8_BYTE + i
            ids = self._merge_tokens(ids, best_pair, new_id)
            merges[best_pair] = new_id

            if verbose:
                print(f"Training merge {i+1}/{n_merges}: {best_pair} -> {new_id}")
        
        self.bp_merges = merges
        self.vocab = self._get_vocab()
    
    def save(self, path: str) -> None:
        """Save tokenizer."""
        
        with open(path, 'w') as f:
            f.write(f"{self.vocab_size}\n")
            f.write(f"{self.special_tokens}\n")
            f.write(f"{self.bp_merges}\n")
    
    def load(self, path: str) -> None:
        """Load tokenizer."""
        
        with open(path, 'r') as f:
            self.vocab_size = int(f.readline())
            self.special_tokens = eval(f.readline())
            self.bp_merges = eval(f.readline())

        self.vocab = self._get_vocab()


class RegExTokenizer(BaseTokenizer):
    """Tokenizer using a regex pattern to initially split text before tokenization."""
    def __init__(self, vocab_size: int = 800, pattern: str = SPLIT_PATTERN, special_tokens: Dict[str, int] = {}):
        super().__init__(vocab_size)
        self.split_pattern = pattern
        self.re_pattern = re.compile(self.split_pattern)

        self.special_tokens = special_tokens
        self.inv_special_tokens = {v: k for k, v in special_tokens.items()}

    def train(self, text: str, verbose: bool = False) -> None:
        """Train tokenizer."""
        UTF8_BYTE = 256
        
        # Tokenize text using the vocab and a utf-8 encoding
        re_splits = re.findall(self.re_pattern, text) # Initially split text based on regex pattern
        ids = [list(split.encode('utf-8')) for split in re_splits]

        n_merges = self.vocab_size - UTF8_BYTE - len(self.special_tokens)
        merges = {}
        
        # Iterate through the token pairs and merge them
        for i in range(n_merges):
            
            stats = {}
            for split in ids:
                stats.update(self._get_stats(split)) # Update stats each regex split at a time
            
            if not stats:
                break
                
            best_pair = max(stats, key=stats.get) # Find most common pair                                                   

            new_id = UTF8_BYTE + i                                                       
            ids = [self._merge_tokens(split, best_pair, new_id) for split in ids]                               
            merges[best_pair] = new_id

            if verbose:
                print(f"Training merge {i+1}/{n_merges}: {best_pair} -> {new_id}")
            
        self.bp_merges = merges
        self.vocab = self._get_vocab() 

    def encode(self, text: str) -> List[int]:
        """Produce a list of tokens from a text string. Handles special tokens."""
        
        if len(self.special_tokens) == 0:
            return self._encode_text(text)
        
        special_pattern = '(' + '|'.join([re.escape(token) for token in self.special_tokens]) + ')'
        special_splits = re.split(special_pattern, text)

        ids = []
        for split in special_splits:
            if split in self.special_tokens:
                ids.append(self.special_tokens[split])
            else:
                ids.extend(self._encode_text(split))

        return ids

    def _encode_text(self, text: str) -> List[int]:
        """Produce a list of tokens from a text string."""
        
        # Tokenize text using the vocab and a utf-8 encoding
        re_splits = re.findall(self.re_pattern, text)

        ids = []
        for split in re_splits:
            ids.extend(self._encode_split(split))
        
        return ids

    def _encode_split(self, split: str) -> List[int]:
        """Encode a single split of text."""
        
        ids = [id for id in split.encode('utf-8')]
        while True:
            
            if len(ids) < 2:
                break

            stats = self._get_stats(ids)
            merge_pair = min(stats, key=lambda x: self.bp_merges.get(x, float('inf')))

            if merge_pair not in self.bp_merges:
                break

            new_id = self.bp_merges[merge_pair]
            ids = self._merge_tokens(ids, merge_pair, new_id)

        return ids

    def decode_list(self, tokens: List[int]) -> str:
        """Decode list of tokens to a text string."""
        
        byte_list = []
        for token in tokens:
            if token in self.inv_special_tokens:
                byte_list.append(self.inv_special_tokens[token].encode('utf-8'))
            elif token in self.vocab:
                byte_list.append(self.vocab[token])
            else:
                raise ValueError(f"Token {token} not found in vocabulary.")
        
        return [byte.decode('utf-8', errors='replace') for byte in byte_list]

    def decode(self, tokens: List[int]) -> List[str]:
        """Decode list of tokens to a list of strings."""
        return ''.join(self.decode_list(tokens))
    

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a BPE tokenizer.')
    parser.add_argument('-v', '--vocab_size', type=int, default=8000, help='Size of the vocabulary.')
    parser.add_argument('-t', '--train', action='store_true', help='Train tokenizer.')
    parser.add_argument('-d', '--train_data', type=str, default=None, help='Path to training data.')
    parser.add_argument('-s', '--save', type=str, default=None, help='Path to tokenizer save file.')
    parser.add_argument('-l', '--load', type=str, default=None, help='Path to tokenizer load file.')
    parser.add_argument('--special', type=str, default=None, help='Path to special tokens file.')
    parser.add_argument('--test_str', type=str, default=None, help='Test string to encode.')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode.')
    args = parser.parse_args()

    if args.train:
        assert args.save is not None, "Must provide a save path when training."
        # assert args.train_data is not None, "Must provide a training data path when training."
    else:
        assert args.load is not None, "Must provide a load path when not training."
        assert args.test_str is not None, "Give me a test string to encode if not training!"

    return args


def main(args: argparse.Namespace):
    """Execute tokenizer training or testing."""
    special_tokens = {}
    if args.special is not None:
        with open(args.special, 'r') as f:
            for i, line in enumerate(f):
                token = line.strip()
                token_id = args.vocab_size - i - 1
                special_tokens[token] = token_id

    tokenizer = RegExTokenizer(vocab_size=args.vocab_size, special_tokens=special_tokens)

    if args.train:
        with open('data/labels.txt', 'r') as f:
            text = f.read()[:5000000]
        
        tokenizer.train(text, verbose=args.verbose)
        tokenizer.save(args.save)
    else:
        tokenizer.load(args.load)
        tokens = tokenizer.encode(args.test_str)

        print(f'Length of test string: {len(args.test_str)}')
        print(f'Number of tokens: {len(tokens)}')
        print(f'Compression ratio: {len(args.test_str) / len(tokens):.2f}x')
        print('')
        print(f"Encoded tokens: {tokens}")
        decoded = tokenizer.decode_list(tokens)
        print(f"Decoded string: {decoded}")

        # Join decoded together and omit all whitespace
        decoded_str = ''.join(decoded).replace(' ', '')
        print(f"Output: {decoded_str}")


if __name__ == "__main__":

    args = parse_args()
    main(args)