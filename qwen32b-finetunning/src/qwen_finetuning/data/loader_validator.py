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
