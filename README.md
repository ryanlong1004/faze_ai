# Faze AI

## Ongoing Questions

- For continuous training
  - Do we deploy these as `crawlers`, like web crawlers only buckets intead of websites
  - Do we need a database for v1?  Saving the pickled encodings to the customer bucket is pretty secure.
  - Open browser windows for display

## Overview

Faze AI is a face recognition tool that allows you to encode known faces, recognize faces in images, and validate unknown faces using a command-line interface (CLI).

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

### Encoding Known Faces

To encode known faces from the `training` directory:

```
python src/cli.py encode --model <model> --encodings-location <path_to_encodings>
```

- `--model`: Model to use for face recognition (default: "hog")
- `--encodings-location`: Path to save/load encodings (default: "output/encodings.pkl")

### Recognizing Faces

To recognize faces in images from a specified directory:

```
python src/cli.py recognize <images_path> --model <model> --encodings-location <path_to_encodings>
```

- `images_path`: Path to the directory containing images
- `--model`: Model to use for face recognition (default: "hog")
- `--encodings-location`: Path to save/load encodings (default: "output/encodings.pkl")

### Validating Unknown Faces

To validate unknown faces from the `unknown` directory:

```
python src/cli.py validate_faces --model <model>
```

- `--model`: Model to use for face recognition (default: "hog")

## Directory Structure

```
/Users/rlong/Sandbox/faze_ai/
├── src/
│   ├── cli.py
│   ├── main.py
├── training/
├── output/
├── validation/
├── unknown/
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License.
