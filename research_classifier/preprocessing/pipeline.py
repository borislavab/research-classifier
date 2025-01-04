from typing import List
from .processors import AbstractProcessor, DefaultTokenizer
from .dataset import extract_labels


class Pipeline:
    def __init__(self, processors: List[AbstractProcessor]):
        self.processors = processors

    def process_abstract(self, abstract: str) -> str:
        for processor in self.processors:
            abstract = processor.process(abstract)
        return abstract


def process_sample(sample: dict) -> dict:
    pipeline = Pipeline([DefaultTokenizer()])
    sample["abstract"] = pipeline.process_abstract(sample["abstract"])
    sample["labels"] = extract_labels(sample["categories"])
    return sample
