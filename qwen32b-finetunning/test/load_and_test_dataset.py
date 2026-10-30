from qwen_finetuning.models.model_loader import (
    get_model,
)

from qwen_finetuning.models.tokenizer import (
    get_tokenizer,
)

from qwen_finetuning.models.lora_config import (
    get_lora_config,
)


def main():

    print("Loading tokenizer...")

    tokenizer = get_tokenizer()

    print("Tokenizer OK")

    print("Loading Qwen3-32B 4-bit...")

    model = get_model()

    print("Model OK")

    print(
        "Model class:",
        model.__class__.__name__
    )

    print(
        "Device:",
        model.device
    )

    print(
        "LoRA config:"
    )

    lora_config = get_lora_config()

    print(lora_config)


if __name__ == "__main__":
    main()