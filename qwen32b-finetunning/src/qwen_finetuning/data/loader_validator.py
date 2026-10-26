from pathlib import Path
from datasets import load_dataset, DatasetDict


VALID_ROLES = {
    "system",
    "user",
    "assistant"
}

def get_dataset(dataset_name: str) -> DatasetDict:
    """
    Download the complete dataset, including all available splits.
    """
    return load_dataset(dataset_name)

 


def validate_dataset(example):
    
    messages = example.get("messages")
    
    if not isinstance(messages, list):
        return False
    
    if len(messages) == 0:
        return False
    
    has_user = False
    has_assistant = False
       
    
    for messages in messages:
        role = messages.get("role")
        content = messages.get("content")
        print(role,"\n")
        print(content, "\n")
        
        
            
        if role not in VALID_ROLES:
            return False
        
        if not isinstance(content, str):
            return False
        
        if role == "user":
            has_user = True

        if role == "assistant":
            has_assistant = True
            
    return has_user, has_assistant
