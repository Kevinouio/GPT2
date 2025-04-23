import os
import time
import math
import torch
import torch.nn.functional as F
from model.gpt2 import GPT
from model.config import GPTConfig
from .scheduler import get_lr
from .optimizer import configure_optimizers
from .utils import setup_ddp, is_master_process, GradScalerWrapper
from data.dataloader import DataLoaderLite
from tokenizer.tokenizer import get_tokenizer


def train():
    device, ddp, ddp_rank, ddp_local_rank, ddp_world_size = setup_ddp()
    enc = get_tokenizer()
    torch.set_float32_matmul_precision('high')

    # Data setup
    B = 64
    T = 1024
    total_batch_size = 524288
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    # Model setup
    config = GPTConfig(vocab_size=50304)
    model = GPT(config).to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    optimizer = configure_optimizers(raw_model, weight_decay=0.1, learning_rate=6e-4, device_type=device.type)

    # Training setup
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073

    scaler = GradScalerWrapper(device_type=device.type)
    log_file = "logs/log.txt"
    os.makedirs("logs", exist_ok=True)
    open(log_file, "w").close()

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            scaler.backward(loss)

        if ddp:
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        scaler.step(optimizer)
        scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt

        if is_master_process():
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")


if __name__ == "__main__":
    train()
