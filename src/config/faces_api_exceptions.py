from http import HTTPStatus

from config import swagger_strings as sws
from config.exceptions import APIException


class InvalidBase64CodeException(APIException):
    message = sws.DOC_ERR_INVALID_BASE64
    http_code = HTTPStatus.BAD_REQUEST


class HealthTestsMissingException(APIException):
    message = sws.DOC_ERR_NO_HEALTH_TESTS
    http_code = HTTPStatus.NOT_FOUND


class ClientExistsException(APIException):
    message = sws.DOC_ERR_CLIENT_EXISTS
    http_code = HTTPStatus.FORBIDDEN


class ClientNotExistsException(APIException):
    message = sws.DOC_ERR_CLIENT_NOT_EXISTS
    http_code = HTTPStatus.NOT_FOUND


class InvalidBaseDatasetException(APIException):
    message = sws.DOC_ERR_INVALID_BASE_DATASET
    http_code = HTTPStatus.BAD_REQUEST


class InvalidClientIDHeaderException(APIException):
    message = sws.DOC_ERR_INVALID_CLIENT_HEADER
    http_code = HTTPStatus.BAD_REQUEST


class InvalidPersonIDException(APIException):
    message = sws.DOC_ERR_INVALID_PERSON_ID
    http_code = HTTPStatus.BAD_REQUEST


class PersonNotExistsException(APIException):
    message = sws.DOC_ERR_PERSON_NOT_EXISTS
    http_code = HTTPStatus.NOT_FOUND


class DataServiceUnavailableException(APIException):
    message = sws.DOC_ERR_DATA_SERVICE_UNAVAILABLE
    http_code = HTTPStatus.SERVICE_UNAVAILABLE


class TFSServiceUnavailableException(APIException):
    message = sws.DOC_ERR_TFS_SERVICE_UNAVAILABLE
    http_code = HTTPStatus.SERVICE_UNAVAILABLE


class BadBatchLengthForGroupingRecognition(APIException):
    message = sws.DOC_ERR_BAD_LENGTH_ON_FACES_BATCH_FOR_GROUPING
    http_code = HTTPStatus.BAD_REQUEST
