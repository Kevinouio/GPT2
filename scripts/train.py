# scripts/train.py
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
# from torch.nn.parallel import DistributedDataParallel as DDP # Placeholder for DDP
# from torch.distributed import init_process_group, destroy_process_group # Placeholder for DDP

# Model and Config
# Add project root to sys.path to allow imports from config, src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import TrainConfig, gpt2_small, gpt2_medium, gpt2_large, gpt2_xl # Import model size functions
from src.model.gpt2 import GPT, GPTConfig

# Tokenizer
from src.data.tokenizer import enc, vocab_size # Import chosen tokenizer instance and vocab_size

# -----------------------------------------------------------------------------
# Configuration Loading & Setup
# -----------------------------------------------------------------------------

# Load default train config
train_cfg = TrainConfig()

# --- Overrides from command line (Example: python train.py --device=cpu --compile=False) ---
# Basic CLI argument parsing (can be replaced with argparse for more complex needs)
config_keys = [k for k,v in vars(train_cfg).items()]
for arg in sys.argv[1:]:
    if arg.startswith('--'):
        keyval = arg.split('=', 1)
        if len(keyval) == 2:
            key, value = keyval
            key = key[2:] # remove '--'
            if key in config_keys:
                try:
                    # Attempt to eval it (e.g. for bools, ints, floats)
                    attempt = eval(value)
                except (SyntaxError, NameError):
                    # If eval fails, assume it's a string
                    attempt = value
                # Ensure the type matches the config's type
                if isinstance(attempt, type(getattr(train_cfg, key))):
                    print(f"Overriding: {key} = {attempt}")
                    setattr(train_cfg, key, attempt)
                else:
                     print(f"Warning: Type mismatch for {key}. Expected {type(getattr(train_cfg, key))}, got {type(attempt)}. Ignoring override.")
            else:
                print(f"Warning: Unknown config key: {key}")
        else:
            print(f"Warning: Malformed argument: {arg}")

# --- Choose Model Size ---
model_configs = {
    'gpt2': gpt2_small,
    'gpt2-medium': gpt2_medium,
    'gpt2-large': gpt2_large,
    'gpt2-xl': gpt2_xl,
}
if train_cfg.model_type in model_configs:
    model_cfg_builder = model_configs[train_cfg.model_type]
    print(f"Using model type: {train_cfg.model_type}")
else:
    print(f"Warning: Unknown model_type '{train_cfg.model_type}'. Defaulting to 'gpt2' (small).")
    model_cfg_builder = gpt2_small

# --- Setup Device & Precision ---
if train_cfg.device == 'cuda':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print("WARNING: CUDA not available, forcing CPU.")
        device = 'cpu'
        train_cfg.device = 'cpu' # Update config to reflect reality
else:
     device = 'cpu'

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[train_cfg.dtype]

# Check if selected dtype is supported
if train_cfg.dtype == 'bfloat16' and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
    print("WARNING: bfloat16 not supported on this device/setup. Forcing float32.")
    train_cfg.dtype = 'float32'
    ptdtype = torch.float32
elif train_cfg.dtype == 'float16' and not torch.cuda.is_available():
    print("WARNING: float16 requires CUDA. Forcing float32.")
    train_cfg.dtype = 'float32'
    ptdtype = torch.float32

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
print(f"Using device: {device}, dtype: {train_cfg.dtype}")
# Note: We don't set default device globally here, will move data explicitly

# --- Create Checkpoint Directory ---
if not os.path.exists(train_cfg.out_dir):
    os.makedirs(train_cfg.out_dir)

# -----------------------------------------------------------------------------
# Data Loading & Preparation
# -----------------------------------------------------------------------------
data_dir = 'data'
train_data_path = os.path.join(data_dir, 'train.bin')
val_data_path = os.path.join(data_dir, 'val.bin')

# Attempt to use tiktoken tokenizer first
if enc is None or vocab_size <= 0:
     raise RuntimeError("Tokenizer (enc) could not be initialized. Check src/data/tokenizer.py and ensure data exists if using CharTokenizer.")

# Check if pre-tokenized data exists, otherwise create it
input_file_path = os.path.join(data_dir, 'input.txt') # Assumed raw data file
if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
    if not os.path.exists(input_file_path):
         raise FileNotFoundError(
             f"Error: Raw data file '{input_file_path}' not found. "
             f"Needed to create tokenized binary files ('{train_data_path}', '{val_data_path}'). "
             f"Please place your raw text data in '{input_file_path}'."
         )
    print(f"Tokenizing data from {input_file_path}...")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    n = len(data)
    train_data_text = data[:int(n*0.9)]
    val_data_text = data[int(n*0.9):]

    # Encode the text data using the loaded tokenizer
    # Ensure encode returns a list or numpy array of integers
    train_ids_list = enc.encode(train_data_text)
    val_ids_list = enc.encode(val_data_text)

    print(f"Training data has {len(train_ids_list):,} tokens")
    print(f"Validation data has {len(val_ids_list):,} tokens")

    # Export to bin files
    # Use np.uint16 for memory efficiency if vocab_size < 65536
    dtype_np = np.uint16 if vocab_size < 65536 else np.int32
    print(f"Using numpy dtype: {dtype_np} for saving token data.")
    train_ids = np.array(train_ids_list, dtype=dtype_np)
    val_ids = np.array(val_ids_list, dtype=dtype_np)
    train_ids.tofile(train_data_path)
    val_ids.tofile(val_data_path)
    print(f"Tokenized data saved to '{train_data_path}' and '{val_data_path}'.")
else:
    print(f"Loading pre-tokenized data from '{train_data_path}' and '{val_data_path}'...")
    # Determine dtype from vocab size for loading
    dtype_np = np.uint16 if vocab_size < 65536 else np.int32

# Load tokenized data using memory mapping
train_data = np.memmap(train_data_path, dtype=dtype_np, mode='r')
val_data = np.memmap(val_data_path, dtype=dtype_np, mode='r')
print(f"Loaded train data: {len(train_data):,} tokens")
print(f"Loaded val data: {len(val_data):,} tokens")

# --- Data Batching Function ---
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Ensure block_size doesn't exceed data length
    max_start_index = len(data) - train_cfg.block_size
    if max_start_index <= 0:
        raise ValueError(f"Dataset split '{split}' is too small ({len(data)} tokens) for block_size {train_cfg.block_size}.")

    ix = torch.randint(max_start_index, (train_cfg.batch_size,))
    # Retrieve sequences and convert to int64 immediately for PyTorch compatibility
    x = torch.stack([torch.from_numpy((data[i:i+train_cfg.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+train_cfg.block_size]).astype(np.int64)) for i in ix])

    # Move data to the target device
    if device_type == 'cuda':
        # Pin memory for faster CPU -> GPU transfer
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Model Initialization / Resuming
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

# Determine model configuration arguments
model_args = dict(n_layer=0, n_head=0, n_embd=0, block_size=train_cfg.block_size,
                  bias=False, vocab_size=50304, dropout=0.0) # Default values, will be updated

if train_cfg.init_from == 'scratch':
    # Init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = model_cfg_builder() # Get config dataclass instance
    gptconf.vocab_size = vocab_size # Set the correct vocab size from tokenizer
    gptconf.block_size = train_cfg.block_size # Set block size from train config
    model = GPT(gptconf)
    model_args = vars(gptconf) # Store the actual args used
elif train_cfg.init_from == 'resume':
    print(f"Resuming training from {train_cfg.out_dir}")
    ckpt_path = os.path.join(train_cfg.out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}. Cannot resume.")
    checkpoint = torch.load(ckpt_path, map_location=device) # Load to target device directly

    # Load model config from checkpoint - IMPORTANT for consistency
    checkpoint_model_args = checkpoint['model_config']
    # Force these args even if config changed, ensures compatible model loading
    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)
    model_args = vars(gptconf) # Store args from checkpoint

    state_dict = checkpoint['model']
    # Fix potential state dict keys issues (e.g., from DDP, compile)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    # Load optimizer state - will be done after optimizer creation
    optimizer_state = checkpoint.get('optimizer', None) # Use .get for backward compat
    print(f"Resumed from iteration {iter_num}, best validation loss: {best_val_loss:.4f}")
    # train_cfg = checkpoint['train_config'] # Option: Restore entire train config

elif train_cfg.init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {train_cfg.init_from}")
    from transformers import GPT2LMHeadModel # Requires transformers library
    # Ensure the chosen model type matches the HuggingFace model
    if train_cfg.model_type != train_cfg.init_from:
         print(f"Warning: train_cfg.model_type ('{train_cfg.model_type}') does not match init_from ('{train_cfg.init_from}'). Using '{train_cfg.init_from}' for config.")
    try:
        # Get the corresponding config builder based on the HF model name
        hf_model_name = train_cfg.init_from
        if hf_model_name not in model_configs:
            raise ValueError(f"No local config builder found for HuggingFace model '{hf_model_name}'. Available: {list(model_configs.keys())}")
        gptconf = model_configs[hf_model_name]() # Build our config
        gptconf.vocab_size = vocab_size # Use our tokenizer's vocab size
        gptconf.block_size = train_cfg.block_size
        model_args = vars(gptconf)

        print(f"Loading weights from HuggingFace model: {hf_model_name}")
        model_hf = GPT2LMHeadModel.from_pretrained(hf_model_name)
        sd_hf = model_hf.state_dict()

        # Initialize our model
        model = GPT(gptconf)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # Ignore this mask/buffer

        # Copy weights carefully, handling potential key name differences and transpositions
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys) == len(sd_hf.keys()), f"Our model keys ({len(sd_keys)}) != HF model keys ({len(sd_hf.keys())})"

        for k in sd_keys:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for specific layers, transpose required
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
                    print(f"Copied and transposed: {k} | HF Shape: {sd_hf[k].shape} -> Our Shape: {sd[k].shape}")
            else:
                # Vanilla copy
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    print(f"Copied: {k} | Shape: {sd[k].shape}")

        print("Successfully loaded weights from HuggingFace model.")

    except ImportError:
        print("ERROR: `transformers` library not installed. Cannot initialize from GPT-2 weights.")
        print("Install it with: pip install transformers")
        exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize from HuggingFace model '{train_cfg.init_from}': {e}")
        exit(1)

else:
     raise ValueError(f"Unknown init_from value: {train_cfg.init_from}. Must be 'scratch', 'resume', or 'gpt2*'.")

# Move model to target device
model.to(device)

# Print model details
print(f"Model Vocab Size: {model.config.vocab_size}, Block Size: {model.config.block_size}")
print(f"Number of parameters: {model.get_num_params()/1e6:.2f}M")

# -----------------------------------------------------------------------------
# Optimizer & Scaler Setup
# -----------------------------------------------------------------------------
# Create optimizer AFTER model is initialized and potentially loaded
optimizer = model.configure_optimizers(train_cfg.weight_decay, train_cfg.learning_rate,
                                       (train_cfg.beta1, train_cfg.beta2), device_type)

# Load optimizer state if resuming
if train_cfg.init_from == 'resume' and optimizer_state is not None:
    print("Loading optimizer state...")
    try:
        optimizer.load_state_dict(optimizer_state)
        print("Optimizer state loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load optimizer state: {e}. Starting with fresh optimizer state.")
        # Optionally reset optimizer state here if loading failed critically

# Initialize Gradient Scaler for float16 AMP
scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg.dtype == 'float16'))
print(f"Gradient Scaler enabled: {scaler.is_enabled()}")

# -----------------------------------------------------------------------------
# Model Compilation (Optional)
# -----------------------------------------------------------------------------
if train_cfg.compile:
    if hasattr(torch, 'compile'):
        print("Compiling the model... (this might take a minute)")
        try:
            model = torch.compile(model) # Requires PyTorch 2.0+
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Proceeding without compilation.")
            train_cfg.compile = False # Disable if failed
    else:
        print("torch.compile not available (requires PyTorch 2.0+). Skipping compilation.")
        train_cfg.compile = False

# -----------------------------------------------------------------------------
# Learning Rate Scheduler
# -----------------------------------------------------------------------------
def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < train_cfg.warmup_iters:
        return train_cfg.learning_rate * it / train_cfg.warmup_iters
    # 2) If it > lr_decay_iters, return min learning rate
    if it > train_cfg.lr_decay_iters:
        return train_cfg.min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - train_cfg.warmup_iters) / (train_cfg.lr_decay_iters - train_cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return train_cfg.min_lr + coeff * (train_cfg.learning_rate - train_cfg.min_lr)

# -----------------------------------------------------------------------------
# Evaluation Function
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(train_cfg.eval_iters, device=device) # Store losses on target device
        for k in range(train_cfg.eval_iters):
            X, Y = get_batch(split)
            # Use the AMP context manager during evaluation as well
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item() # Get mean loss as a Python float
    model.train() # Set model back to training mode
    return out

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
X, Y = get_batch('train') # Fetch the very first batch to init timings etc.
t0 = time.time()
local_iter_num = 0 # iterations run in this script
raw_model = model.module if hasattr(model, 'module') else model # unwrap DDP/compile
running_mfu = -1.0

print(f"\nStarting training loop for {train_cfg.max_iters} iterations...")
while True:

    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if train_cfg.decay_lr else train_cfg.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % train_cfg.eval_interval == 0 :
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # Wandb logging (optional)
        # if wandb_log:
        #     wandb.log({
        #         "iter": iter_num,
        #         "train/loss": losses['train'],
        #         "val/loss": losses['val'],
        #         "lr": lr,
        #         "mfu": running_mfu*100, # convert to percentage
        #     })
        if losses['val'] < best_val_loss or train_cfg.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0: # Don't save checkpoint at step 0 before any training
                checkpoint = {
                    'model': raw_model.state_dict(), # Save unwrapped model state
                    'optimizer': optimizer.state_dict(),
                    'model_config': model_args, # Use the actual args the model was created with
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'train_config': vars(train_cfg), # Save training config state
                }
                ckpt_path = os.path.join(train_cfg.out_dir, 'ckpt.pt')
                print(f"Saving checkpoint to {ckpt_path} (val_loss: {best_val_loss:.4f})")
                torch.save(checkpoint, ckpt_path)

    # Termination condition based on max iterations
    if iter_num > train_cfg.max_iters:
        print(f"Reached max iterations ({train_cfg.max_iters}). Stopping training.")
        break

    # --- Training Step ---
    # Use a loop for gradient accumulation
    optimizer.zero_grad(set_to_none=True) # Reset grads at the start of the accumulation cycle
    for micro_step in range(train_cfg.gradient_accumulation_steps):
        # Get batch - ensures gradients are accumulated over different data batches
        X, Y = get_batch('train')

        # Forward pass with Automatic Mixed Precision context
        with ctx:
            logits, loss = model(X, Y)
            # Scale loss for accumulation - IMPORTANT!
            loss = loss / train_cfg.gradient_accumulation_steps

        # Backward pass with gradient scaling for fp16
        # We retain graph ONLY if it's NOT the last micro-step, AND if DDP is used (placeholder)
        # retain_graph = (micro_step < train_cfg.gradient_accumulation_steps - 1) # Simplified check
        # scaler.scale(loss).backward(retain_graph=retain_graph)
        scaler.scale(loss).backward() # Simpler: let it handle graph retention if needed internally


    # Clip gradients (after backward pass, before optimizer step)
    if train_cfg.grad_clip > 0.0:
        scaler.unscale_(optimizer) # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

    # Step the optimizer and update the scaler
    scaler.step(optimizer)
    scaler.update()

    # Timing and Logging
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    t0 = t1 # update start time for next iteration
    if iter_num % train_cfg.log_interval == 0:
        # Get loss as float. Note: this is the scaled loss for the last micro-step
        # To get the approx total loss for the full batch, multiply by accumulation steps
        lossf = loss.item() * train_cfg.gradient_accumulation_steps
        # Calculate MFU (Model Flops Utilization) - Optional but useful
        if local_iter_num >= 5: # let lightning strike 5 times for more accurate timing
             mfu = raw_model.estimate_mfu(train_cfg.batch_size * train_cfg.gradient_accumulation_steps, dt)
             running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6e}, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1


# --- End of Training ---
print("\nTraining finished.")

# Save final checkpoint
final_ckpt_path = os.path.join(train_cfg.out_dir, 'ckpt_final.pt')
print(f"Saving final checkpoint to {final_ckpt_path}")
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_config': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss, # Log the best val loss achieved
    'train_config': vars(train_cfg),
}
torch.save(checkpoint, final_ckpt_path)

print("Done.")