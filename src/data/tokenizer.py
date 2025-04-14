# tokenizer.py
import tiktoken
import os
import pickle
import torch

# --- Using TikToken (Recommended) ---
def get_tiktoken_tokenizer(model_name="gpt2"):
    """Returns a tiktoken tokenizer."""
    enc = tiktoken.get_encoding(model_name)
    return enc

# --- Example: Simple Character-Level Tokenizer (for educational purposes) ---
class CharTokenizer:
    def __init__(self, text=None, cache_path='char_tokenizer_cache.pkl'):
        self.chars = []
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
        self.cache_path = cache_path

        if os.path.exists(cache_path):
            self.load_from_cache()
            print(f"Loaded character tokenizer from {cache_path}")
        elif text:
            self.build_vocab(text)
            self.save_to_cache()
            print(f"Built character tokenizer and saved to {cache_path}")
        else:
            print("Warning: Initializing empty CharTokenizer. Provide text or ensure cache exists.")


    def build_vocab(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi] # Skip unknown chars

    def decode(self, l):
        if isinstance(l, torch.Tensor):
            l = l.tolist()
        return ''.join([self.itos[i] for i in l if i in self.itos]) # Skip unknown indices

    def save_to_cache(self):
         data = {
             'chars': self.chars,
             'stoi': self.stoi,
             'itos': self.itos,
             'vocab_size': self.vocab_size
         }
         with open(self.cache_path, 'wb') as f:
             pickle.dump(data, f)

    def load_from_cache(self):
        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)
        self.chars = data['chars']
        self.stoi = data['stoi']
        self.itos = data['itos']
        self.vocab_size = data['vocab_size']


# enc = get_tiktoken_tokenizer()
# vocab_size = enc.n_vocab # Update config.py if using tiktoken

# Example usage for CharTokenizer (needs data)
# with open('data/input.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
# char_enc = CharTokenizer(text)
# vocab_size = char_enc.vocab_size # Update config.py
# enc = char_enc # Use this instance elsewhere


enc = get_tiktoken_tokenizer()
vocab_size = enc.n_vocab
print(f"Using TikToken tokenizer (gpt2). Vocab size: {vocab_size}")