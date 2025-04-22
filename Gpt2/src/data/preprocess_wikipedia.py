# data/preprocess_wikipedia.py
"""
Wikipedia dataset preprocessor for GPT training.
Downloads, tokenizes, and chunks the dataset into shards.
Usage:
$ python preprocess_wikipedia.py
Output: saves token shards in `wiki_tokens/`
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
LOCAL_DIR = "wiki_tokens"
SHARD_SIZE = int(1e8)  # 100M tokens/shard

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), LOCAL_DIR)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# -----------------------------
# Tokenizer Setup
# -----------------------------
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token out of uint16 bounds"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# -----------------------------
# Load + Process Dataset
# -----------------------------
print("Loading Wikipedia dataset...")
dataset = load_dataset("wikipedia", "20220301.en", split="train")

nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, dataset, chunksize=16):
        if token_count + len(tokens) < SHARD_SIZE:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"wikipedia_{split}_{shard_index:06d}")
            remainder = SHARD_SIZE - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            leftover = len(tokens) - remainder
            all_tokens_np[0:leftover] = tokens[remainder:]
            token_count = leftover

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"wikipedia_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
