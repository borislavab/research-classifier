from research_classifier.preprocessing.categories import calculate_label_counts
import numpy as np
from datasets import Dataset
from research_classifier.training.dataset import load
import sys


def calculate_drop_prob(count: int, median: int) -> float:
    if count <= median:
        return 0.0
    return (count - median) / count


def filter_sample(sample: dict, drop_probs: dict) -> bool:
    for category in sample["categories"].split(" "):
        if np.random.random() > drop_probs[category]:
            # keep sample since for one category it should not be dropped
            return True
    # if all categories say to drop, drop the sample
    return False


# Implement basic undersampling where classes with more samples than the median are undersamples
# with a probability that makes their count equal to the median
# for samples with multiple categories, remove them with the joint probability of removing each category
# as a result combinations are more likely to be preserved
# TODO: implement more sophisticated undersampling which reasons about unnecessary samples to remove
# e.g. near-miss or condensed nearest neighbor
# new dataset length is 1250081
def undersample(dataset: Dataset, threshold: float = None) -> Dataset:
    if threshold is None:
        # find median of category counts
        category_counts = calculate_label_counts(dataset)
        threshold = np.median(list(category_counts.values()))
        print(f"Using median {threshold} as threshold")
    drop_probs = {
        category: calculate_drop_prob(count, threshold)
        for category, count in category_counts.items()
    }
    return dataset.filter(lambda sample: filter_sample(sample, drop_probs))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python undersampling.py <dataset_path> <output_path>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    dataset = load(dataset_path=dataset_path)
    undersampled = undersample(dataset=dataset)
    undersampled.to_json(output_path)
    print(len(undersampled))
