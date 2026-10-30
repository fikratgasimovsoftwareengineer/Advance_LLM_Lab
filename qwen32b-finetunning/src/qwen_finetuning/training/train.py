import os
## get dataset

from qwen_finetuning.data.loader_validator import get_dataset
from qwen_finetuning.models.model_loader import get_model
from qwen_finetuning.models.lora_config import get_lora_config
from qwen_finetuning.models.tokenizer import get_tokenizer
from qwen_finetuning.training.finetunning import get_trainer


def main():
    
    dataset = get_dataset("trl-lib/Capybara")
    
    print(f"Train samples {len(dataset['train'])}")
    print(f"Test samples {len(dataset['test'])}")
    tokenizer = get_tokenizer()
    
    if tokenizer is None:
        return False
    
    model = get_model()
    if model is None:
        return False
    
    #lora
    lora_config = get_lora_config()
    
    trainer = get_trainer(model=model,
                          tokenizer=tokenizer,
                          dataset = dataset,
                          lora_config = lora_config)
    
    print(trainer.model.print_trainable_parameters())
    
    trainer.train()
    
    trainer.save_model(
        "/checkpoints/qwen32b/final_adapter"
    )
    
if __name__ == "__main__":
    main()