import kagglehub

from src.config.datasets import DATASETS

for dataset in DATASETS:
    path = kagglehub.dataset_download(dataset)
    print(f"Path to dataset files for {dataset}: {path}")
