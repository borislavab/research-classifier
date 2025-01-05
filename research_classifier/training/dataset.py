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


def preprocess(dataset: Dataset):
    # TODO: process in batches: batched=True, batch_size=1000
    return dataset.map(
        pipeline.process_sample, remove_columns=["title", "categories", "abstract"]
    )


def split_dataset(dataset: Dataset):
    return dataset.train_test_split(test_size=0.01, shuffle=True, seed=42)


def undersample(dataset: Dataset):
    pass


def oversample_smote(dataset: Dataset):
    pass


def load_for_training() -> Tuple[Dataset, Dataset]:
    dataset = load()
    processed = preprocess(dataset)
    split = split_dataset(processed)
    return split["train"], split["test"]


if __name__ == "__main__":
    # file_path = download_dataset()
    dataset_path = "/Users/bbagaliyska/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/212/"
    dataset_path += "/arxiv-metadata-oai-snapshot.json"
    dataset = load(dataset_path, head=5)
    processed = preprocess(dataset)
    for sample in processed:
        print(sample)
    print("-----------------")
    split = split_dataset(processed)
    for sample in split["train"]:
        print(sample)
    print("-----------------")
    for sample in split["test"]:
        print(sample)
