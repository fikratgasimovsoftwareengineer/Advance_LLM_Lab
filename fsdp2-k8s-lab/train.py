import os, time, torch
import torch.distributed as dist
from datasets import load_dataset
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import(
    get_model_state_dict,   
    StateDictOptions
)
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET_NAME = os.getenv("DATASET_NAME", "tatsu-lab/alpaca")
DATASET_SAMPLES = int(os.getenv("DATASET_SAMPLES", "512"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
EPOCHS = min(int(os.getenv("EPOCHS", "1")), 5)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))
LR = float(os.getenv("LR", "2e-5"))

def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def fmt(x):
    instruction = x.get("instruction", "")
    inp = x.get("input", "")
    output = x.get("output", "")
    if inp:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {"text": text}

def main():
    rank, world_size, local_rank = setup()
    device = torch.device("cuda", local_rank)

    print(f"[rank={rank}] gpu={torch.cuda.get_device_name(local_rank)} world_size={world_size}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(DATASET_NAME, split="train")
    ds = ds.select(range(min(DATASET_SAMPLES, len(ds))))
    ds = ds.map(fmt)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collator, pin_memory=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    mesh = init_device_mesh("cuda", (world_size,))
    mp = MixedPrecisionPolicy(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float16,
    )

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp)
            
    fully_shard(model, mesh=mesh, mp_policy=mp)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    opt.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        start = time.time()
        
        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            
            loss = out.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if rank == 0 and step % 20 == 0:
                alloc = torch.cuda.memory_allocated(device) / (1024**3)
                print(f"epoch={epoch+1}/{EPOCHS} step={step}/{len(loader)} loss={out.loss.item():.4f} gpu={alloc:.2f}GiB", flush=True)

        dist.barrier()
        
        state_dict = get_model_state_dict(
                    model,
                    options = StateDictOptions(
                        full_state_dict=True,
                        cpu_offload=True,
                    ),
                
                )
        if rank == 0:
            print(f"EPOCH {epoch+1}/{EPOCHS} COMPLETED in {time.time()-start:.1f}s", flush=True)
            path = f"/checkpoints/model_epoch_{epoch+1}.pt"
            
            torch.save(
                state_dict,
                path
            )
            print(
            f"Checkpoint salvato in {path}",
                flush=True
            )
           

    dist.barrier()
    if rank == 0:
        print("TRAINING COMPLETED", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
