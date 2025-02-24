import logging
import os

import numpy as np
from grpc import RpcError, insecure_channel
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from config.faces_api_exceptions import TFSServiceUnavailableException
from config.settings import (FLASK_APP_NAME, TF_SERVING_NOISE_MODEL_NAME,
                             TF_SERVING_NOISE_MODEL_SIGNATURE_NAME,
                             TFS_HOST_ENV, TFS_PORT_ENV, TFS_REQUESTS_TIMEOUT)
from logic.utils.tf_utils import handle_tfs_error


def get_noise_predictions(image: np.array) -> float:
    channel = insecure_channel(f"{os.getenv(TFS_HOST_ENV)}:{os.getenv(TFS_PORT_ENV)}")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = TF_SERVING_NOISE_MODEL_NAME
    request.model_spec.signature_name = TF_SERVING_NOISE_MODEL_SIGNATURE_NAME
    request.inputs["images"].CopyFrom(make_tensor_proto(image))

    try:
        response = stub.Predict(request, TFS_REQUESTS_TIMEOUT)
    except RpcError as e:
        handle_tfs_error(e, request.model_spec.name, logging.getLogger(FLASK_APP_NAME))
        raise TFSServiceUnavailableException()

    return response.outputs["scores"].float_val
