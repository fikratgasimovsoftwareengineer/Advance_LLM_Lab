import torch
from transformers import AutoModelForCausalLM
from qwen_finetuning.models.quantizazione import (
   get_quantizazione_config
)


MODEL_NAME = "Qwen/Qwen3-14B"

def get_model():
    
    quantization_config = (
        get_quantizazione_config()
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype = torch.bfloat16,
        low_cpu_memory_usage="True" 

    )
    model.config.use_cache = False
    
    return model


