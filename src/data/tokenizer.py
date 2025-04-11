import tiktoken
import torch


def encode(text):
    encoder = tiktoken.get_encoding("r50k_base")
    return encoder.encode(text)

def decode(text):
    decoder = tiktoken.get_encoding("r50k_base")
    return decoder.decode(text)


