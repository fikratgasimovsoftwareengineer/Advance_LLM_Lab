import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)

dist.init_process_group(
    backend="nccl",
    device_id=torch.device(f"cuda:{local_rank}")
)

world_size = dist.get_world_size()
rank = dist.get_rank()

mesh = init_device_mesh(
    "cuda",
    (world_size,)   
)

print(
    f"rank={rank}, "
    f"local_rank={local_rank}, "
    f"world_size={world_size}"
)

print(mesh)

dist.destroy_process_group()