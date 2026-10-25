import os
import socket

import torch
import torch.distributed as dist

from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


class MiniModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 128)
        
    def forward(self, x):
        
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
    
    
def main():
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    device = torch.device(
        f"cuda: {local_rank}"
    )
    
    dist.init_process_group(
        backend="nccl",
        device_id=device
    )
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    pid = os.getpid()
    socket_nm = socket.gethostname()
    
    print(  f"\n"
        f"PID        = {pid}\n"
        f"HOST       = {socket_nm}\n"
        f"RANK       = {rank}\n"
        f"LOCAL_RANK = {local_rank}\n"
        f"WORLD_SIZE = {world_size}\n"
        f"GPU        = {torch.cuda.get_device_name(local_rank)}\n",
        flush = True)
    
    ### init device mesh  ###
    mesh = init_device_mesh(
        "cuda"
    )