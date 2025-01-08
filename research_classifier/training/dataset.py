from typing import Tuple
from datasets import Dataset
from transformers import DataCollatorWithPadding
from research_classifier.preprocessing import (
    DefaultTokenizer,
    Pipeline,
    load,
)

tokenizingProcessor = DefaultTokenizer(is_training=True)
# start simple with no other preprocessing than tokenization to see results
# motivation is that the abstracts are in natural language
# which matches BERT's pretraining inputs
# and are likely to be high quality information-rich text
# a good candidate for future preprocessing is handling special characters in formulas
# iterate later to see what works best in practice
pipeline = Pipeline(tokenizingProcessor)
collator = DataCollatorWithPadding(tokenizer=tokenizingProcessor.tokenizer)


def preprocess(dataset: Dataset, batch_size: int = 1000):
    return dataset.map(
        pipeline.process_sample,
        batched=True,
        batch_size=batch_size,
        remove_columns=["categories", "abstract"],
    )


def split_dataset(dataset: Dataset):
    # evaluation is currently a bottleneck - decrease test size
    return dataset.train_test_split(test_size=0.005, shuffle=True, seed=42)


def load_for_training(
    dataset_path: str = None, head: int = None
) -> Tuple[Dataset, Dataset]:
    dataset = load(dataset_path, head=head)
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
