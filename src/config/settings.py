"""Project settings."""

import os

# API SETTINGS
CELEBRITES_MODEL_HOST_ENV = "CELEBRITES_MODEL_HOST"
CELEBRITES_MODEL_PORT_ENV = "CELEBRITES_MODEL_PORT"
FACES_PATH_ENV = "FACES_PATH"
DEFAULT_LOG_LEVEL = "ERROR"  # ERROR, INFO, DEBUG
LOG_LEVEL_ENV = "LOG_LEVEL"
TFS_REQUESTS_TIMEOUT = 10  # in secs
DATA_REQUESTS_TIMEOUT = 20  # in secs

# API VERSION
FACES_API_VERSION = "VERSION_NUMBER"

# MODEL FILES
CELEBRITES_MODEL_HOST = os.getenv(CELEBRITES_MODEL_HOST_ENV, "172.17.0.2")
CELEBRITES_MODEL_PORT = int(os.getenv(CELEBRITES_MODEL_PORT_ENV, "8000"))
CELEBRITES_NAMES_PATH = os.path.join(os.getenv(FACES_PATH_ENV), "models/celebs/lfw_msceleb_names_ids_6m.joblib")

# MODEL PARAMETERS
FACE_DETECTION_THRESHOLD = [0.6, 0.7, 0.7]  # these are thresholds for the three stages of detection
DEFAULT_RECOGNITION_THRESHOLD = 0.85

# The following field is the default value for the threshold for recognition. That is, each distance lower than this
# value between the predicted vector in the celebrity model and the query vector is a MATCH for the model; and
# the others are a NOT MATCH.
DEFAULT_CELEBRITY_RECOGNITION_THRESHOLD = 0.25

# This coefficient is an adjustment to normalize distances in the output of the celebrity model and the faces model.
# SO, the maximum distance for each model means the same confidence to the API user.
DEFAULT_CELEBRITY_RECOGNITION_COEFFICIENT = 3.4  # RECOGNITION_THRESHOLD / CELEBRITY_RECOGNITION_THRESHOLD

LOWER_RECOGNITION_THRESHOLD = 0.7  # faces as close as this are not worth saving if another exists
UNKNOWN_FACES_LIMIT = int(os.getenv("UNKNOWN_FACES_LIMIT", 5))  # max num. of faces automatically trained for an unknown

EMBEDDING_SIZE = 128
DEFAULT_FACE_IMAGE_SIZE = 160

NOISE_DETECTION_THRESHOLD = 0.9  # Higher threshold implies being more restrictive with noise but also lose more faces.

# distance used on the clustering algorithm for face recognition when grouping parameter is set to true.
DEFAULT_CLUSTERING_RECOGNITION_THRESHOLD = 0.85

# FLASK
FLASK_APP_NAME = os.getenv("FLASK_APP_NAME")
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 10336

# DATA API
DATA_API_HOST_ENV = "DATA_HOST"
DATA_API_PORT_ENV = "DATA_PORT"
DATA_API_VERSION_ENV = "DATA_VERSION"

# TENSORFLOW SERVING
TFS_HOST_ENV = "TFS_HOST"
TFS_PORT_ENV = "TFS_PORT"
TFS_EMBEDDINGS_OPERATION = "calculate_embeddings"
TFS_MODEL_NAME = "faces_recognition"

# NOISE SCORE
TF_SERVING_NOISE_MODEL_NAME = "noise"
TF_SERVING_NOISE_MODEL_SIGNATURE_NAME = "serving_default"

# EMOTION
TF_SERVING_EMOTION_MODEL_NAME = "emotion_recognition"
TF_SERVING_EMOTION_MODEL_SIGNATURE_NAME = "predict_images"
TF_SERVING_EMOTION_MODEL_INPUT = "inputs"
TF_SERVING_EMOTION_MODEL_OUTPUT = "scores"
EMOTION_LABELS_DATASET = "fer2013"

# STATSD
STATSD_ADDRESS_ENV = "STATSD_ADDRESS"
STATSD_BATCH_SIZE_ENV = "STATSD_BATCH_SIZE"
STATSD_APP_PREFIX_ENV = "STATSD_APP_PREFIX"
DEFAULT_STATSD_ADDRESS = "localhost:8125"
DEFAULT_STATSD_BATCH_SIZE = 100
