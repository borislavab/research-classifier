from typing import List
from .processors import AbstractProcessor, AbstractTokenizer, DefaultTokenizer
from .dataset import extract_labels


class Pipeline:
    def __init__(
        self,
        tokenizer: AbstractTokenizer,
        processors: List[AbstractProcessor] = None,
    ):
        self.tokenizer = tokenizer
        self.processors = processors if processors else []

    def process_abstract(self, abstract: str) -> str:
        for processor in self.processors:
            abstract = processor.process(abstract)
        return self.tokenizer.tokenize(abstract)

    def process_sample(self, sample: dict) -> dict:
        tokenized_sample = self.process_abstract(sample["abstract"])
        tokenized_sample["labels"] = extract_labels(sample["categories"])
        return tokenized_sample
