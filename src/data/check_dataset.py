from datasets import load_dataset

print("Connecting to clapAI/MultiLingualSentiment...")

dataset = load_dataset(
    "clapAI/MultiLingualSentiment",
    streaming=True
)

print("\nDataset loaded successfully.")
print(dataset)

for split in dataset:
    print(f"\nSplit: {split}")
    print("First row:")
    print(next(iter(dataset[split])))