from typing import List
from .abstract import AbstractProcessor, AbstractTokenizer
from .categories import CategoriesProcessor
import numpy as np


class Pipeline:
    """
    Pipeline which takes care of the full preprocessing of samples
    both in training and prediction time.
    Flexible design which allows toggling and reordering different steps
    to find which sequence works best for the model.
    Preprocessing can easily be extended with additional functionalities
    by implementing new AbstractProcessor subclasses.
    """

    def __init__(
        self,
        tokenizer: AbstractTokenizer,
        abstract_processors: List[AbstractProcessor] = None,
    ):
        self.tokenizer = tokenizer
        self.abstract_processors = abstract_processors if abstract_processors else []
        self.categories_processor = CategoriesProcessor()

    def process_abstract(self, abstract: str | List[str]) -> str | List[str]:
        for processor in self.abstract_processors:
            if isinstance(abstract, str):
                abstract = processor.process(abstract)
            else:
                abstract = [processor.process(a) for a in abstract]
        return self.tokenizer.tokenize(abstract)

    def process_categories(self, categories: str | List[str]) -> np.ndarray:
        return self.categories_processor.process(categories)

    def process_sample(self, sample: dict) -> dict:
        tokenized_sample = self.process_abstract(sample["abstract"])
        tokenized_sample["labels"] = self.process_categories(sample["categories"])
        return tokenized_sample

    def get_labels(self):
        return self.categories_processor.get_labels()
