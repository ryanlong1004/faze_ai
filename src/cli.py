import logging
from pathlib import Path

import click

from main import (DEFAULT_ENCODINGS_PATH, encode_known_faces, recognize_faces,
                  validate)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@click.command()
@click.option("--model", default="hog", help="Model to use for face recognition")
@click.option(
    "--encodings-location", default=DEFAULT_ENCODINGS_PATH, type=click.Path(), help="Path to save/load encodings"
)
def encode(model, encodings_location):
    encode_known_faces(model=model, encodings_location=Path(encodings_location))


@click.command()
@click.argument("images_path", type=click.Path(exists=True))
@click.option("--model", default="hog", help="Model to use for face recognition")
@click.option(
    "--encodings-location", default=DEFAULT_ENCODINGS_PATH, type=click.Path(), help="Path to save/load encodings"
)
def recognize(images_path, model, encodings_location):
    recognize_faces(images_path=images_path, model=model, encodings_location=Path(encodings_location))


@click.command()
@click.option("--model", default="hog", help="Model to use for face recognition")
def validate_faces(model):
    validate(model=model)


cli.add_command(encode)
cli.add_command(recognize)
cli.add_command(validate_faces)

if __name__ == "__main__":
    cli()
