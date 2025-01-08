from datasets import Dataset
from research_classifier.preprocessing.dataset import load
from research_classifier.analysis.scrape_categories import scraped_categories
from research_classifier.preprocessing.categories import get_labels
import sys
import ollama
import random
from typing import List
import json

prompt_template = """
You are a scientist writing a research paper.
Generate a short abstract of an article on the topic of "{category_friendly_name}".
Please include only the abstract text. Make sure its about 3-4 sentences with maximal total length of 300 words.
Here is an example for reference:
"{reference_abstract}"
"""


def create_prompts(dataset: Dataset, tag: str, category_friendly_name: str, count: int):
    samples = dataset.filter(lambda x: tag in x["categories"].split(" "))
    prompts = []
    for i in range(count):
        reference_abstract = random.choice(samples)["abstract"].strip()
        prompt = prompt_template.format(
            category_friendly_name=category_friendly_name,
            reference_abstract=reference_abstract,
        )
        print(prompt)
        prompts.append(prompt)
    return prompts


def oversample_llama(dataset: Dataset, labels: List[int] = None, target_count=1):
    """oversample the dataset to the target count for each label"""

    # make it more creative
    options = {"temperature": 0.7}
    category_tags = get_labels(dataset)
    category_names = [scraped_categories.get(tag, tag) for tag in category_tags]
    samples = []

    for label in labels:
        prompts = create_prompts(
            dataset, category_tags[label], category_names[label], target_count
        )
        for prompt in prompts:
            response = ollama.generate(model="llama3.2", prompt=prompt, options=options)
            generated_abstract = response.response.strip()
            # if it's in quotes, remove them
            if generated_abstract.startswith('"'):
                generated_abstract = generated_abstract[1:]
            if generated_abstract.endswith('"'):
                generated_abstract = generated_abstract[:-1]
            samples.append(
                {"categories": [category_tags[label]], "abstract": generated_abstract}
            )

    return samples


def store_samples(samples, output_path):
    with open(output_path, "w") as f:
        # write in jsonl format
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "Usage: python oversampling.py <dataset_path> <output_path> <target_count> <labels>"
        )
        sys.exit(1)
    dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    target_count = int(sys.argv[3])
    labels = [int(label) for label in sys.argv[4:]]

    dataset = load(dataset_path=dataset_path)
    generated_samples = oversample_llama(dataset, labels, target_count)
    store_samples(generated_samples, output_path)
