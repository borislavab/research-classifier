from transformers import pipeline
from datasets import Dataset
from research_classifier.analysis.categories import (
    distinct_categories_count as category_counts,
)
from research_classifier.preprocessing.dataset import load
from research_classifier.analysis.scrape_categories import scraped_categories
import sys


def oversample_gpt(dataset: Dataset):
    oversample_counts = {
        category: min(500, 1000 - count)
        for category, count in category_counts.items()
        if count < 1000
    }
    print(len(oversample_counts))
    oversample_counts = {
        k: v for k, v in oversample_counts.items() if k not in category_map
    }
    print(len(oversample_counts))
    print(oversample_counts)
    # generator = pipeline("text-generation", model="gpt2")
    # prompt = "Write a research paper abstract on the topic of "
    # for category, count in oversample_counts.items():
    #     if category not in scraped_categories:
    #         print(f"Category {category} not found")
    # print(scraped_categories[category])
    # category_prompt = f"{category}, "


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python oversampling.py <dataset_path> <output_path>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    # dataset = load(dataset_path=dataset_path)
    # oversampled = oversample_gpt(dataset=dataset)
    oversample_gpt(
        dataset=Dataset.from_dict({"categories": ["math.OC", "math.OC", "math.OC"]})
    )
    # oversampled.to_json(output_path)
    # print(len(oversampled))
