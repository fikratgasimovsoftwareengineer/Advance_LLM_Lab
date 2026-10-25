from datasets import load_dataset

DATASET_NAME = "trl-lib/Capybara"

def load_data():
    dataset = load_dataset(DATASET_NAME,
                           split="train")
    print(dataset)
    """_
    [
        {
            "role":"user", "content":"...",
            
        },
        {
            "role":"assistant", "content":"..."
        }
    ]
    """
if __name__ =="__main__":
    load_data()