import os
import socket
import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    
    
    pid = os.getpid()
    hostname = socket.gethostname()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(rank)
    
    gpu_name = torch.cuda.get_device_name(local_rank)
    print(
        
        
        f"PID={pid} |"
        f"HOST = {hostname} |"
        f"RANK= {rank} |"
        f"LOCAL_RANK= {local_rank} |"
        f"WORLD_SIZE= {world_size} | "
        f"GPU= {gpu_name}",
        flush=True
    )
    
    dist.barrier()
    dist.destroy_process_group()
    
if __name__ == "__main__":
    main()