import kagglehub

from training.datasets import DATASETS


def main():
    for dataset in DATASETS:
        path = kagglehub.dataset_download(dataset, path=None, force_download=True)
        print(f"Path to dataset files for {dataset}: {path}")


if __name__ == "__main__":
    main()
