from __future__ import absolute_import, division, print_function

import logging
import os
from typing import Dict, List

import numpy as np
from grpc import RpcError, insecure_channel
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from config.faces_api_exceptions import TFSServiceUnavailableException
from config.settings import (FLASK_APP_NAME, TFS_HOST_ENV, TFS_PORT_ENV,
                             TFS_REQUESTS_TIMEOUT)
from logic.utils.tf_utils import handle_tfs_error


def get_face_gender_predictions(images: np.array) -> List[Dict[str, float]]:
    channel = insecure_channel(f"{os.getenv(TFS_HOST_ENV)}:{os.getenv(TFS_PORT_ENV)}")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    results = []

    for image_data in images:
        result = {}
        request = predict_pb2.PredictRequest()
        request.model_spec.name = "gender_recognition"
        request.model_spec.signature_name = "predict_images"

        request.inputs["images"].CopyFrom(make_tensor_proto(image_data))

        try:
            response = stub.Predict(request, TFS_REQUESTS_TIMEOUT)
        except RpcError as e:
            handle_tfs_error(e, request.model_spec.name, logging.getLogger(FLASK_APP_NAME))
            raise TFSServiceUnavailableException()

        for i, label in enumerate(["male", "female"]):
            result[label] = round(response.outputs["scores"].float_val[i], 2)

        results.append(result)

    return results
