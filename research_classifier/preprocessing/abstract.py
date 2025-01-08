from abc import ABC, abstractmethod
from nltk.corpus import stopwords
import nltk
from transformers import AutoTokenizer
import torch
from nltk.stem import WordNetLemmatizer


class AbstractProcessor(ABC):
    """
    Goal of this abstraction is to be able to easily tweak preprocessing steps for abstracts
    to find which works best for the BERT model.
    A future goal could be to also persist the processors used in training along with the model,
    so the same preprocessing pipeline can be used for inference.
    """

    # TODO: support batch processing
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


class Lemmatizer(AbstractProcessor):
    def __init__(self):
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()

    def process(self, abstract: str) -> str:
        return " ".join([self.lemmatizer.lemmatize(word) for word in abstract.split()])


class AbstractTokenizer(ABC):
    @abstractmethod
    def tokenize(self, abstract: str):
        pass


class DefaultTokenizer(AbstractTokenizer):
    def __init__(self, is_training: bool = False):
        self.is_training = is_training
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", is_fast=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def tokenize(self, abstract: str):
        # do not pad at this stage to max length,
        # instead use dynamic padding to max sequence length in batch
        # when batching with data collator
        #
        # truncation should be good enough as abstracts are rarely longer than 512 tokens
        # and even when they are, they don't exceed by much
        if self.is_training:
            # do not return pytorch tensors in training, it led to error
            return self.tokenizer(abstract, truncation=True)
        else:
            return self.tokenizer(abstract, truncation=True, return_tensors="pt").to(
                self.device
            )
