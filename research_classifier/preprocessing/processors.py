from abc import ABC, abstractmethod
from nltk.corpus import stopwords
import nltk


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
