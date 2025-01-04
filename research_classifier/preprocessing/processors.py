from abc import ABC, abstractmethod
from nltk.corpus import stopwords
import nltk
from transformers import AutoTokenizer


class AbstractProcessor(ABC):

    @abstractmethod
    def process(self, abstract: str) -> str:
        pass


class StopWordRemover(AbstractProcessor):
    def __init__(self):
        nltk.download("stopwords")

    def process(self, abstract: str) -> str:
        return " ".join(
            [
                word.strip()
                for word in abstract.split()
                if word.strip().lower() not in stopwords.words("english")
                and len(word.strip()) > 0
            ]
        )


class DefaultTokenizer(AbstractProcessor):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", is_fast=True)

    def process(self, abstract: str) -> str:
        # do not pad at this stage to max length,
        # instead use dynamic padding to max sequence length in batch
        # when batching with data collator
        #
        # truncation should be good enough as abstracts are rarely longer than 512 tokens
        # and even when they are, they don't exceed by much
        return self.tokenizer(abstract, truncation=True)
