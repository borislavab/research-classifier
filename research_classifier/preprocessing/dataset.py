import pandas as pd
from typing import List
import kagglehub
import os
from datasets import load_dataset, Dataset
import numpy as np


def download():
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    print("Path to dataset files:", path)
    return os.path.join(path, "arxiv-metadata-oai-snapshot.json")


def load(dataset_path: str = None, head: int = None) -> Dataset:
    if not dataset_path:
        dataset_path = download()
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.select_columns(["categories", "abstract"])
    if head:
        return dataset.select(range(head))
    return dataset
