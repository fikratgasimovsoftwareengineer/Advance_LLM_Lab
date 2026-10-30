from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-32B"

def get_tokenizer():
    
    """
    Load the tokenizer and chat template associated
    with Qwen3-32B.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer