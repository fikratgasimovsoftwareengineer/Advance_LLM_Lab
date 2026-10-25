from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
from torch.distributed.checkpoint.state_dict import(
    get_model_state_dict,
    StateDictOptions
)

from torch.optim.optimizer import lo

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT = "/data/checkpoints/model_epoch_3.pt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

state_dict=torch.load(
    CHECKPOINT,
    map_location="cpu"
)
model.load_state_dict(state_dict)
model.to(device)

model.eval()
print("Modello caricato correttamente")


### INFERENCE ###

prompt = "Spiega cos'e il machine learning in modo semplice"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_token=150
    )
    
print(tokenizer.decode(
    outputs[0],
    skip_special_tokens= True
))