# Build a base tokenizer class for tokenizing text data

import re
from typing import List, Dict, Tuple

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

    def decode(self, tokens: List[int]):
        """Decode list of tokens to a text string."""
        return b''.join([self.vocab[token] for token in tokens]).decode('utf-8')
    
    def decode_list(self, tokens: List[int]):
        """Decode list of tokens to a list of strings."""
        return [self.vocab[token].decode('utf-8') for token in tokens]
    
    def train(self, text: str, verbose: bool = False):
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
    
    def save(self, path: str):
        """Save tokenizer."""
        raise NotImplementedError
    
    def load(self, path: str):
        """Load tokenizer."""
        raise NotImplementedError
    


if __name__ == "__main__":

    tokenizer = BaseTokenizer(vocab_size=1000)
    
    print('Reading Tokenizer training data...')

    with open('final_png_formulas.txt', 'r') as f:
        text = f.read()[:1000000]

    print('Read complete.')

    print('Training Tokenizer...')
    tokenizer.train(text, verbose=True)

    test_str = r"\left( \mathcal { N } _ { 1 } e ^ { - \xi M _ { 1 } \xi - \xi \lambda _ { 1 } } \right) * \left( \mathcal { N } _ { 2 } e ^ { - \xi M _ { 2 } \xi - \xi \lambda _ { 2 } } \right) = \mathcal { N }"
    tokens = tokenizer.encode(test_str)
    
    print(test_str)
    print(tokens)
    print(tokenizer.decode_list(tokens))
