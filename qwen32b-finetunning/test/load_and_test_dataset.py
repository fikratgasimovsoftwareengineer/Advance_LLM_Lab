from qwen_finetuning.data.loader_validator import (
    get_dataset,
    validate_dataset,
)


def main():
    DATASET_NAME = "trl-lib/Capybara"
    # 1. Download / load Capybara
    dataset = get_dataset(DATASET_NAME)

    print("\n==============================")
    print("RAW DATASET")
    print("==============================")
    print(dataset)

    # 2. Validation
    validated_data = validate_dataset(dataset)

    print("\n==============================")
    print("VALIDATED DATASET")
    print("==============================")

    for split_name, split in validated_data.items():
        print(
            f"{split_name}: {len(split)} samples"
        )


if __name__ == "__main__":
    main()