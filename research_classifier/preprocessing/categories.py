from typing import List
from datasets import Dataset
import numpy as np
from research_classifier.analysis.categories import distinct_categories
from research_classifier.preprocessing.dataset import download, load

# these categories should be semantically identical
# but the official category according to arxiv taxonomy
# should be the one on the right
# see https://arxiv.org/category_taxonomy
category_map = {
    "cmp-lg": "cs.CL",
    "funct-an": "math.FA",
    "supr-con": "cond-mat.supr-con",
    "bayes-an": "physics.data-an",
    "acc-phys": "physics.acc-ph",
    "comp-gas": "nlin.CG",
    "adap-org": "nlin.AO",
    "ao-sci": "physics.ao-ph",
    "chem-ph": "physics.chem-ph",
    "atom-ph": "physics.atom-ph",
    "plasm-ph": "physics.plasm-ph",
    "patt-sol": "nlin.PS",
    "dg-ga": "math.DG",
    "mtrl-th": "cond-mat.mtrl-sci",
}


def extract_categories(dataset: Dataset) -> List[str]:
    all_categories = set()
    for categories in dataset["categories"]:
        all_categories.update(categories.split(" "))
    return sorted(all_categories)


def get_labels(dataset: Dataset = None):
    all_categories = distinct_categories
    if dataset:
        all_categories = extract_categories(dataset)
    mapped = {category_map.get(category, category) for category in all_categories}
    return sorted(mapped)


def extract_sample_categories(categories: str) -> List[str]:
    return {category_map.get(category, category) for category in categories.split(" ")}


class CategoriesProcessor:
    def __init__(self):
        self.labels = get_labels()
        self.label_count = len(self.labels)

    def get_labels(self):
        return self.labels

    # supports single sample or batch
    def process(self, categories: str | List[str]) -> np.ndarray:
        # Handle single string case
        if isinstance(categories, str):
            sample_categories = extract_sample_categories(categories)
            return np.array(
                [1.0 if label in sample_categories else 0.0 for label in self.labels]
            )

        # Handle batch case
        sample_category_sets = [
            extract_sample_categories(sample_categories)
            for sample_categories in categories
        ]
        sample_count = len(categories)

        output = np.zeros((sample_count, self.label_count), dtype=np.float32)

        for i, label in enumerate(self.labels):
            output[:, i] = [
                1.0 if label in sample_categories else 0.0
                for sample_categories in sample_category_sets
            ]

        return output


if __name__ == "__main__":
    path = download()
    dataset = load(path)
    all_categories = extract_categories(dataset)
    print(all_categories)
