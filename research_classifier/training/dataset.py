from typing import Tuple
from datasets import Dataset
from transformers import DataCollatorWithPadding
from research_classifier.preprocessing import (
    DefaultTokenizer,
    Pipeline,
    load,
)

tokenizingProcessor = DefaultTokenizer()
pipeline = Pipeline(tokenizingProcessor)
collator = DataCollatorWithPadding(tokenizer=tokenizingProcessor.tokenizer)


def preprocess(dataset: Dataset, batch_size: int = 1000):
    return dataset.map(
        pipeline.process_sample,
        batched=True,
        batch_size=batch_size,
        remove_columns=["title", "categories", "abstract"],
    )


def split_dataset(dataset: Dataset):
    return dataset.train_test_split(test_size=0.01, shuffle=True, seed=42)


def load_for_training(dataset_path: str = None) -> Tuple[Dataset, Dataset]:
    dataset = load(dataset_path)
    processed = preprocess(dataset)
    split = split_dataset(processed)
    return split["train"], split["test"]


if __name__ == "__main__":
    # file_path = download_dataset()
    dataset_path = "/Users/bbagaliyska/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/212/"
    dataset_path += "/arxiv-metadata-oai-snapshot.json"
    dataset = load(dataset_path, head=5)
    processed = preprocess(dataset, batch_size=3)
    for sample in processed:
        print(sample)
    print("-----------------")
    split = split_dataset(processed)
    for sample in split["train"]:
        print(sample)
    print("-----------------")
    for sample in split["test"]:
        print(sample)
