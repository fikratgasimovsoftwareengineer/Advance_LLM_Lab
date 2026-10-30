from peft import LoraConfig

def get_lora_config():
    return LoraConfig(
        r = 16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )
    
"""
    Qwen3-32B
↓
4-bit
↓
base frozen
+
LoRA adapters
↓
trainable
"""