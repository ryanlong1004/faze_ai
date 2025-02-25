# detector.py

import logging
import pickle
from collections import Counter
from pathlib import Path

import face_recognition
from joblib import Parallel, delayed
from PIL import Image, ImageDraw

import cli as _cli

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    logger.info("Encoding known faces")

    def process_image(filepath):
        if not filepath.is_file():
            return None, None
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        return name, face_encodings

    filepaths = list(Path("training").rglob("*/*"))
    results = Parallel(n_jobs=-1)(delayed(process_image)(filepath) for filepath in filepaths)

    names = []
    encodings = []
    for name, face_encodings in [result for result in results if result is not None]:
        if name is not None and face_encodings:
            for encoding in face_encodings:
                names.append(name)
                encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)
    logger.info("Finished encoding known faces")


def recognize_faces(
    images_path: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    for image_location in Path(images_path).rglob("*"):
        if not image_location.is_file():
            continue
        logger.info("Recognizing faces in image: %s", image_location)
        with encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)

        input_image = face_recognition.load_image_file(image_location)

        input_face_locations = face_recognition.face_locations(input_image, model=model)
        input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

        pillow_image = Image.fromarray(input_image)
        draw = ImageDraw.Draw(pillow_image)

        for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
            name = _recognize_face(unknown_encoding, loaded_encodings)
            if not name:
                name = "Unknown"
            _display_face(draw, bounding_box, name)

        del draw
        pillow_image.show()
        logger.info("Finished recognizing faces in image: %s", image_location)


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding, 0.5)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    if votes:
        return votes.most_common(1)[0][0]


BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"


def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )


def validate(model: str = "hog"):
    logger.info("Validating unknown faces")
    for filepath in Path("unknown").rglob("*"):
        if filepath.is_file():
            recognize_faces(images_path=str(filepath.absolute()), model=model)
    logger.info("Finished validating unknown faces")


if __name__ == "__main__":
    _cli.cli()
