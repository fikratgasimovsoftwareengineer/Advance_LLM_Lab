from trl import SFTConfig, SFTTrainer

def get_configuration_config():
    
    fsdp_config = {
        "fsdp_version": 2,
        "fsdp_reshard_after_forward": True,
        "fsdp_auto_wrap_policy":"TRANSFORMER_BASED_WRAP",
        "fsdp_activation_checkpointing":True,
        "fsdp_cpu_ram_efficient_loading":True,
        "fsdp_cpu_offload":False,
        "fsdp_state_dict_type": "SHARDED_STATE_DICT"   
    }
    
    return SFTConfig(
        output_dir="/checkpoints/qwen32b",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_length=1024,
        
        assistant_only_loss=True,
        packing=False,
        
        fp16=True,
        bf16=False,
        fsdp=True,
        fsdp_config=fsdp_config,
        
        eval_strategy="steps",
        eval_steps=100,
        
        save_strategy="steps",
        save_total_limit=3,
        
        logging_steps=10,
        seed=42,
        report_to="none"
    )
    
def get_trainer(model, 
                tokenizer, 
                dataset,
                lora_config):
    
    training_config = get_configuration_config()
    
    trainer = SFTTrainer(
        model = model,
        args = training_config,
        train_dataset=dataset["train"],
        eval_dataset = dataset["test"],
        processing_class=tokenizer,
        peft_config=lora_config
    )
    return trainer