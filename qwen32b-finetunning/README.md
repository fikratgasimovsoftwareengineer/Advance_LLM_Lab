# Qwen3-14B Distributed Fine-Tuning with QLoRA, SFT and FSDP2

This project implements a **distributed supervised fine-tuning pipeline for Qwen3-14B** using:

- Hugging Face Transformers
- TRL `SFTTrainer`
- PEFT / LoRA
- bitsandbytes 4-bit NF4 quantization
- PyTorch FSDP2
- `torchrun`
- NCCL
- AWS EC2 GPU instances

The current lab runs on **two AWS `g4dn.xlarge` instances**, each equipped with one **NVIDIA Tesla T4 (16 GB VRAM)**.

The goal is to validate the complete multi-node fine-tuning workflow before moving to Docker, Amazon ECR and Kubernetes/k3s.

---

## 1. High-Level Architecture

```text
                   Hugging Face Hub
                         |
              +----------+----------+
              |                     |
      trl-lib/Capybara       Qwen/Qwen3-14B
              |                     |
              |                     v
              |              4-bit NF4 loading
              |                     |
              |                 Base model
              |                   frozen
              |                     +
              |                LoRA adapters
              |                 trainable
              |                     |
              +-------------+-------+
                            v
                       TRL SFTTrainer
                            |
                            v
                         FSDP2
                            |
                  torch.distributed
                            |
                           NCCL
                    +-------+--------+
                    |                |
                    v                v
              AWS EC2 Node 0   AWS EC2 Node 1
              rank 0           rank 1
              Tesla T4         Tesla T4
                    |                |
                    +---- NCCL/TCP ---+
                            |
                            v
                  Distributed Fine-Tuning
                            |
                            v
                     LoRA Checkpoints
```

---

## 2. Training Strategy

The project does **not** perform full-parameter fine-tuning.

It uses **QLoRA**:

```text
Qwen3-14B
   |
   v
4-bit NF4 quantization
   |
   v
base model frozen
   |
   v
LoRA adapters added
   |
   v
only LoRA parameters are optimized
```

### Technology responsibilities

| Component | Responsibility |
|---|---|
| TRL `SFTTrainer` | Supervised fine-tuning lifecycle |
| bitsandbytes | 4-bit NF4 quantization |
| PEFT / LoRA | Small trainable adapter parameters |
| QLoRA | 4-bit base model + LoRA |
| FSDP2 | Distributed sharding of model state |
| `torchrun` | Multi-process / multi-node launcher |
| NCCL | GPU communication backend |
| Capybara | Conversational supervised training dataset |

---

## 3. Dataset

The project currently uses:

```text
trl-lib/Capybara
```

Capybara is already structured as a conversational dataset:

```python
{
    "messages": [
        {
            "role": "user",
            "content": "..."
        },
        {
            "role": "assistant",
            "content": "..."
        }
    ]
}
```

The dataset is passed directly to `SFTTrainer`.

Manual tokenization is intentionally avoided. TRL and the Qwen tokenizer handle:

```text
messages
   |
   v
Qwen chat template
   |
   v
tokenization
   |
   v
truncation
   |
   v
padding/collation
   |
   v
labels
   |
   v
SFT
```

---

## 4. Repository Structure

```text
qwen32b-finetunning/
├── configs/
│   ├── data/
│   ├── distributed/
│   ├── model/
│   └── training/
├── Dockerfile
├── finetunning/
├── pyproject.toml
├── README.md
├── requirements.txt
├── src/
│   └── qwen_finetuning/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader_validator.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── lora_config.py
│       │   ├── model_loader.py
│       │   ├── quantizazione.py
│       │   └── tokenizer.py
│       └── training/
│           ├── __init__.py
│           ├── finetunning.py
│           └── train.py
└── test/
    ├── __init__.py
    └── load_and_test_dataset.py
```

---

## 5. Model Configuration

Current target model:

```text
Qwen/Qwen3-14B
```

Qwen3-14B was selected instead of Qwen3-32B because the current lab uses two Tesla T4 GPUs with 16 GB VRAM each.

Example quantization configuration:

```python
import torch
from transformers import BitsAndBytesConfig

def get_quantizazione_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.float16,
    )
```

For Tesla T4, the project uses **FP16 compute**.

---

## 6. LoRA Configuration

```python
from peft import LoraConfig

def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
```

The base model remains frozen while the LoRA adapter parameters are trainable.

Before training:

```python
trainer.model.print_trainable_parameters()
```

This is an important guard against accidentally starting full fine-tuning.

---

## 7. FSDP2 Configuration

The project uses FSDP2 through `SFTConfig`.

```python
fsdp_config = {
    "version": 2,
    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    "reshard_after_forward": True,
    "activation_checkpointing": True,
    "cpu_ram_efficient_loading": True,
    "cpu_offload": False,
    "state_dict_type": "SHARDED_STATE_DICT",
}
```

and:

```python
SFTConfig(
    ...
    fsdp=True,
    fsdp_config=fsdp_config,
)
```

### Important checkpointing rule

Do **not** enable both:

```python
gradient_checkpointing=True
```

and:

```python
"activation_checkpointing": True
```

at the same time.

For this project:

```text
Trainer gradient checkpointing   = OFF
FSDP activation checkpointing    = ON
```

---

## 8. Initial Training Configuration

The initial conservative configuration for two Tesla T4 GPUs is:

```text
Model                    Qwen3-14B
Quantization             NF4 4-bit
LoRA rank                16
Per-device batch size    1
Gradient accumulation    8
Sequence length          512
Precision                FP16
FSDP2                    enabled
Activation checkpointing enabled
Reshard after forward    enabled
```

---

## 9. AWS Infrastructure

Current lab topology:

```text
AWS VPC
|
├── Node 0
│   ├── EC2 g4dn.xlarge
│   ├── 1 x Tesla T4 16 GB
│   ├── 16 GiB system RAM
│   ├── private IP: 172.20.26.43
│   └── rank 0
│
└── Node 1
    ├── EC2 g4dn.xlarge
    ├── 1 x Tesla T4 16 GB
    ├── 16 GiB system RAM
    ├── private IP: 172.20.16.151
    └── rank 1
```

The nodes communicate through the AWS private network.

Network interface:

```text
ens5
```

Recommended variables:

```bash
export NCCL_SOCKET_IFNAME=ens5
export GLOO_SOCKET_IFNAME=ens5
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1
```

---

## 10. Distributed Process Topology

```text
nnodes = 2
nproc_per_node = 1
world_size = 2
```

Therefore:

```text
Node 0
rank       = 0
local_rank = 0
GPU        = cuda:0

Node 1
rank       = 1
local_rank = 0
GPU        = cuda:0
```

`rank` identifies the distributed Python process.

---

## 11. Rendezvous

Current endpoint:

```text
MASTER_ADDR = 172.20.26.43
MASTER_PORT = 29500
```

Manual TCP verification:

On Node 0:

```bash
python3 -m http.server 29500 --bind 172.20.26.43
```

On Node 1:

```bash
nc -vz 172.20.26.43 29500
```

Stop the temporary HTTP server before starting `torchrun`.

---

## 12. Launching Distributed Training

Install the package in editable mode on both nodes:

```bash
cd ~/qwen32b-finetunning
python -m pip install -e .
```

### Node 0 / rank 0

```bash
cd ~/qwen32b-finetunning

export NCCL_SOCKET_IFNAME=ens5
export GLOO_SOCKET_IFNAME=ens5
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

torchrun   --nnodes=2   --nproc_per_node=1   --node_rank=0   --master_addr=172.20.26.43   --master_port=29500   --module qwen_finetuning.training.train
```

### Node 1 / rank 1

```bash
cd ~/qwen32b-finetunning

export NCCL_SOCKET_IFNAME=ens5
export GLOO_SOCKET_IFNAME=ens5
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

torchrun   --nnodes=2   --nproc_per_node=1   --node_rank=1   --master_addr=172.20.26.43   --master_port=29500   --module qwen_finetuning.training.train
```

Start rank 0 first, then rank 1.

---

## 13. Checkpoint Directory

The trainer writes to:

```text
/checkpoints/qwen14b
```

Create and authorize it on both nodes:

```bash
sudo mkdir -p /checkpoints/qwen14b
sudo chown -R ubuntu:ubuntu /checkpoints
```

Verify write access:

```bash
touch /checkpoints/test_write
rm /checkpoints/test_write
```

Final adapter:

```text
/checkpoints/qwen14b/final_adapter
```

---

## 14. Storage Requirements

Qwen3-14B requires tens of GB of local disk because the original Hugging Face checkpoint is downloaded before/during quantized loading.

4-bit quantization reduces runtime memory, but it does **not** mean only a 4-bit-sized checkpoint is downloaded.

A development issue encountered was:

```text
No space left on device
```

The root EBS volume was therefore increased.

Recommended root EBS size for this lab:

```text
~150 GiB per node
```

Check with:

```bash
df -h /
```

---

## 15. Hugging Face Cache

Default cache location:

```text
~/.cache/huggingface
```

Inspect:

```bash
du -sh ~/.cache/huggingface
```

Avoid deleting the cache after a successful model download.

---

## 16. Debugging Checklist

### Network

```bash
ping 172.20.26.43
ping 172.20.16.151
nc -vz 172.20.26.43 29500
```

### GPU

```bash
nvidia-smi
```

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Package

```bash
python -c "import qwen_finetuning; print(qwen_finetuning.__file__)"
```

### Processes

```bash
ps -ef | grep -E '[t]orchrun|[t]rain.py'
```

### Rendezvous

```bash
sudo ss -ltnp | grep 29500
```

### Disk

```bash
df -h /
du -sh ~/.cache/huggingface
```

---

## 17. Problems Encountered and Resolved

### Incorrect Transformers argument

Incorrect:

```python
low_cpu_memory_usage=True
```

Correct:

```python
low_cpu_mem_usage=True
```

### Deprecated dtype argument

Use:

```python
dtype=torch.float16
```

instead of the deprecated:

```python
torch_dtype=torch.float16
```

### Disk exhaustion

Error:

```text
No space left on device
```

Root cause:

```text
insufficient EBS root volume capacity
```

Resolution:

```text
increase EBS volume and expand the Linux filesystem
```

### Activation checkpointing conflict

Invalid:

```text
FSDP activation_checkpointing = True
Trainer gradient_checkpointing = True
```

Correct:

```text
FSDP activation_checkpointing = True
Trainer gradient_checkpointing = False
```

### Checkpoint permission error

Error:

```text
PermissionError: [Errno 13] Permission denied: '/checkpoints'
```

Resolution:

```bash
sudo mkdir -p /checkpoints/qwen14b
sudo chown -R ubuntu:ubuntu /checkpoints
```

### Network interface

The AWS interface used by the nodes is:

```text
ens5
```

not `eth0`.

---

## 18. Software Stack

```text
train.py
   |
   v
TRL SFTTrainer
   |
   v
Transformers Trainer
   |
   +--> PEFT / LoRA
   |
   +--> bitsandbytes 4-bit
   |
   v
Accelerate
   |
   v
PyTorch FSDP2
   |
   v
torch.distributed
   |
   v
NCCL
   |
   v
CUDA
   |
   v
Tesla T4
```

---

## 19. Current Project Status

```text
Dataset loading                 ✅
Qwen tokenizer                  ✅
Qwen3-14B model loading         ✅
4-bit quantization              ✅
LoRA configuration              ✅
SFTTrainer integration          ✅
FSDP2 configuration             ✅
2-node rendezvous               ✅
Private AWS networking          ✅
NCCL communication setup        ✅
Distributed training            ✅ in progress
Checkpoint directory            ✅
```

---

## 20. Next Steps

After validating distributed fine-tuning:

```text
1. Distributed inference
   Qwen3-14B + LoRA
   vLLM
   Tensor Parallelism

2. Containerized deployment
   Docker
   Amazon ECR
   EC2 GPU nodes
   Kubernetes / k3s
```

Target future architecture:

```text
TRAINING
Qwen3-14B
  |
  v
QLoRA
  |
  v
SFT
  |
  v
FSDP2
  |
  v
2 x Tesla T4
  |
  v
LoRA adapter

INFERENCE
Qwen3-14B + LoRA
  |
  v
vLLM
  |
  v
Tensor Parallelism
  |
  v
2 x GPU
```

---

## 21. Key Concepts Demonstrated

This lab demonstrates:

- LLM supervised fine-tuning
- PEFT / LoRA / QLoRA
- 4-bit NF4 quantization
- multi-node PyTorch training
- FSDP2
- `torchrun`
- rank / local rank / world size
- NCCL
- AWS VPC private networking
- GPU-enabled EC2
- EBS sizing
- checkpoint management
- distributed debugging
- Hugging Face Transformers / TRL / PEFT / Accelerate

---

## License

This repository is intended for research, experimentation and educational use.

Model and dataset licenses remain subject to the licenses of their respective upstream providers.
