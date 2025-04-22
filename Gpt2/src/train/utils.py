import os
import torch
from torch.distributed import init_process_group, destroy_process_group

def setup_ddp():
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
    return torch.device(device), ddp, rank, local_rank, world_size

def is_master_process():
    return int(os.environ.get('RANK', '0')) == 0

class GradScalerWrapper:
    def __init__(self, device_type):
        self.use_fp16 = (device_type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step(self, optimizer):
        self.scaler.step(optimizer)

    def update(self):
        self.scaler.update()
