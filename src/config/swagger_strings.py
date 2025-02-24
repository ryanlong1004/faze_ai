"""Documentation strings."""

import os

from config.settings import DEFAULT_RECOGNITION_THRESHOLD, FACES_API_VERSION

# API
API_NAME = "GrayMeta Faces API"
API_VERSION = os.getenv(FACES_API_VERSION)
API_DESC = "Face detection and recognition API. Also provides emotion recognition."
FACES_NS_NAME = "faces"
FACES_NS_DESC = "Faces operations."
CLIENTS_NS_NAME = "clients"
CLIENTS_NS_DESC = "Client management operations."
PEOPLE_NS_NAME = "people"
PEOPLE_NS_DESC = "People management operations."
HEALTHZ_NS_NAME = "healthz"
HEALTHZ_NS_DESC = "Service monitoring operations."

# MODELS

# NAMESPACES

# OPERATIONS
DOC_DETECT_NAME = "detect"
DOC_DETECT_DESC = "Detects faces in an image."

DOC_NOISE_NAME = "noise"
DOC_NOISE_DESC = "Returns the noise prediction for a given face."

DOC_GENDER_NAME = "gender"
DOC_GENDER_DESC = "Detects gender in a list of faces images."

DOC_EMOTION_NAME = "emotions"
DOC_EMOTION_DESC = "Detects emotion in a face image."

DOC_RECOGNIZE_NAME = "recognize"
DOC_RECOGNIZE_DESC = (
    "For a set of images, identify the person in each image. Assumes that there is only one face in the image."
)

DOC_RECOGNIZE_ASYNC_NAME = "recognize-async"
DOC_RECOGNIZE_ASYNC_DESC = f"Async version of the faces recognition endpoint. {DOC_RECOGNIZE_NAME}"

DOC_EMBEDDINGS_NAME = "embeddings"
DOC_EMBEDDINGS_DESC = "For a set of images, returns the calculated embeddings for each image."

DOC_CELEBRITIES_NAME = "celebrities"
DOC_CELEBRITIES_DESC = (
    "Identifies a celebrity on a face image. "
    "Assumes that image was cropped with the coordinates returned from the detection operation. "
    "Returns the celebrity id, name and a confidence score between 0 and 1; the recommended "
    "threshold is 0.5"
)

DOC_GET_ID_NAME = "get-id"
DOC_GET_ID_DESC = "Get a new person ID, for use in the learn-faces operation."

DOC_HEALTH_NAME = ""
DOC_HEALTH_DESC = "Returns an HTTP 200 (Success) response if the API is up and running."
DOC_FULL_HEALTH_DESC = "Runs the health check tests (if available) and returns a report of the test results."

# COMMON PARAMETERS
DOC_CLIENT_ID = "The client ID for which this operation will be applied."
DOC_IMAGE_FILE = "The image for which this operation will be applied, encoded in Base64 format."

##############


DOC_LEARN_NAME = "learn"
DOC_LEARN_DESC = "Learn a set of faces for a known person."

DOC_DELETE_IMAGES_NAME = "delete"
DOC_DELETE_IMAGES_DESC = "Delete images by their IDs."

DOC_SHOW_PERSON_FACES_NAME = "show-person-faces"
DOC_SHOW_PERSON_FACES_DESC = "Retrieves the images ids used for recognizing a person."

DOC_UPDATE_PERSON_DESC = "Given its ID, updates a person's name and known status."

DOC_DELETE_PERSON_DESC = "Given its ID, deletes a person."

DOC_CREATE_CLIENT_NAME = "create-client"
DOC_CREATE_CLIENT_DESC = "Create client models."

DOC_REMOVE_CLIENT_NAME = "remove-client"
DOC_REMOVE_CLIENT_DESC = "Remove client models."

DOC_GET_CLIENT_NAME = "remove-client"
DOC_GET_CLIENT_DESC = "Show clients."

# RECOGNIZE PARAMETERS

# RECOGNITION PARAMETERS
DOC_RECOGNITION_THRESHOLD = (
    f"The geometric distance threshold used for matching faces. If not provided, defaults to "
    f"{DEFAULT_RECOGNITION_THRESHOLD}. Bigger values result in less strict matching."
)
DOC_ONLY_UNKNOWNS = "Specifies that recognition should run only against the unknowns subset of trained faces."
DOC_GROUPING = (
    "Specifies if the recognition operation should be done by grouping the faces sent "
    "to the API. If this parameter is set to True, make sure that length of faces' batch is greater or "
    "equal than 3 (as a requirement from the clustering algorithm used internally) if not, an exception "
    "is raised."
)

# LEARN PARAMETERS
DOC_LEARN_IMAGES_FILES = "The images containing the faces to learn. Must contain a single and aligned face."
DOC_LEARN_PERSON_ID = "The person ID to learn for the images."
DOC_LEARN_PERSON_NAME = "The person name to learn for the images."

# DELETE PARAMETERS
DOC_DELETE_IMAGES_IMAGES_ID = "The ID of the image to be deleted."

# SHOW PERSON FACES PARAMETERS
DOC_SHOW_PERSON_FACES_PERSON_ID = "The person ID."

# CREATE CLIENT PARAMETERS
DOC_CREATE_CLIENT_DATASET = "Base dataset for the client."

DOC_CLIENT_ID_HEADER_NAME = "Client-Id"

# ERRORS
DOC_ERR_CLIENT_EXISTS = "Client already exists."
DOC_ERR_CLIENT_NOT_EXISTS = "Client doesn't exist."
DOC_ERR_FILE_EXTENSION = "Invalid file extension."
DOC_ERR_INVALID_BASE64 = "Invalid image format."
DOC_ERR_INVALID_BASE_DATASET = "Invalid base dataset."
DOC_ERR_INVALID_CLIENT_HEADER = f"Invalid or missing client_id header ({DOC_CLIENT_ID_HEADER_NAME})."
DOC_ERR_INVALID_PERSON_ID = "Invalid parameter: person_id must be a string containing an integer."
DOC_ERR_PERSON_NOT_EXISTS = "Person doesn't exist."
DOC_ERR_INVALID_IMAGE_COLORS = "Invalid image format (missing color channels)."
DOC_ERR_NO_HEALTH_TESTS = "API health test cases were not found."
DOC_ERR_UNKNOWN = "Unknown exception."
DOC_ERR_INVALID_UNKNOWN_FACE_DISTANCE = "Invalid face distance for an unknown person."
DOC_ERR_DATA_SERVICE_UNAVAILABLE = "Data service unavailable."
DOC_ERR_TFS_SERVICE_UNAVAILABLE = "TFS service unavailable."
DOC_ERR_BAD_LENGTH_ON_FACES_BATCH_FOR_GROUPING = (
    "The batch does not have enough amount of faces to group (must be at least 3)."
)
