"""
Tokenizer loader using GPT-2's byte pair encoding via `tiktoken`.
Provides an initialized tokenizer for consistent use across training and preprocessing.
"""

import tiktoken

def get_tokenizer(name="gpt2"):
    """Returns a tiktoken tokenizer instance."""
    return tiktoken.get_encoding(name)
