# config/config.py
from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:
    block_size: int = 1024  # Max sequence length (context size)
    vocab_size: int = 50304  # Number of tokens (adjusted for tiktoken gpt2, rounded up) - CHECK YOUR TOKENIZER
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of attention heads
    n_embd: int = 768  # Embedding dimension
    dropout: float = 0.1  # Dropout rate
    bias: bool = True  # True: use bias in Linears and LayerNorms, like GPT-2. False: a bit cleaner and maybe faster


# --- Example Configurations ---

def gpt2_small():
    return GPTConfig()  # Default is like GPT-2 small (124M)


def gpt2_medium():
    return GPTConfig(
        n_layer=24,
        n_head=16,
        n_embd=1024
    )  # ~350M params


def gpt2_large():
    return GPTConfig(
        n_layer=36,
        n_head=20,
        n_embd=1280
    )  # ~774M params


def gpt2_xl():
    return GPTConfig(
        n_layer=48,
        n_head=25,
        n_embd=1600
    )  # ~1.5B params


# --- Configuration for Training ---
@dataclass
class TrainConfig:
    # --- Data ---
    dataset: str = 'openwebtext'  # Example dataset name
    gradient_accumulation_steps: int = 8 * 8  # Used to simulate larger batch sizes
    batch_size: int = 12  # If gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 1024  # Max context length for predictions

    # --- Model ---
    model_type: str = 'gpt2'  # Or 'gpt2-medium', 'gpt2-large', etc. Needs logic to select GPTConfig
    init_from: str = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

    # --- Optimizer ---
    learning_rate: float = 6e-4  # Max learning rate
    max_iters: int = 600000  # Total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # Clip gradients at this value, or disable if == 0.0

    # --- Learning Rate Decay Settings ---
    decay_lr: bool = True  # Whether to decay the learning rate
    warmup_iters: int = 2000  # How many steps to warm up for
    lr_decay_iters: int = 600000  # Should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # Minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # --- System ---
    device: str = 'cuda'  # 'cuda' or 'cpu', default will be set in train.py
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16'
    compile: bool = True  # Use PyTorch 2.0 compile()

    # --- Logging/Saving ---
    eval_interval: int = 2000
    log_interval: int = 10
    eval_iters: int = 200
    always_save_checkpoint: bool = True  # If True, always save a checkpoint after each eval
    out_dir: str = 'checkpoints'
