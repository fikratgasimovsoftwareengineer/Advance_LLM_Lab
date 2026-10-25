import os
import torch
import torch.distributed as dist

from torch import nn
from torch.distributed.fsdp import fully_shard


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


def main():

    # 1. Avvia il distributed environment
    dist.init_process_group("nccl")

    # 2. Ogni processo legge la propria GPU
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    device = torch.device(
        f"cuda:{local_rank}"
    )

    # 3. Crea il modello
    model = TinyModel()

    # 4. Applica FSDP2
    fully_shard(model)

    # 5. Crea optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    # 6. Crea dati fittizi
    x = torch.randn(
        4,
        10,
        device=device
    )

    target = torch.randn(
        4,
        10,
        device=device
    )

    # 7. Forward
    output = model(x)

    # 8. Loss
    loss = torch.nn.functional.mse_loss(
        output,
        target
    )

    # 9. Backward
    loss.backward()

    # 10. Aggiornamento
    optimizer.step()
    optimizer.zero_grad()

    print(
        f"rank={dist.get_rank()} "
        f"loss={loss.item()}"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()