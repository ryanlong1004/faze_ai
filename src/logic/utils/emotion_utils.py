import logging
import os
from typing import Dict, List

import cv2
import numpy as np
from faces_api_utils import rgb2gray
from grpc import RpcError, insecure_channel
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from config.faces_api_exceptions import TFSServiceUnavailableException
from config.settings import (EMOTION_LABELS_DATASET, FLASK_APP_NAME,
                             TF_SERVING_EMOTION_MODEL_INPUT,
                             TF_SERVING_EMOTION_MODEL_NAME,
                             TF_SERVING_EMOTION_MODEL_OUTPUT,
                             TF_SERVING_EMOTION_MODEL_SIGNATURE_NAME,
                             TFS_HOST_ENV, TFS_PORT_ENV, TFS_REQUESTS_TIMEOUT)
from logic.utils.tf_utils import handle_tfs_error


class EmotionUtils:
    @staticmethod
    def get_labels(dataset_name: str) -> Dict[int, str]:
        if dataset_name == "fer2013":
            return {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
        elif dataset_name == "imdb":
            return {0: "woman", 1: "man"}
        elif dataset_name == "KDEF":
            return {0: "AN", 1: "DI", 2: "AF", 3: "HA", 4: "SA", 5: "SU", 6: "NE"}
        else:
            raise Exception("Invalid dataset name")

    @staticmethod
    def preprocess_input(x: np.array) -> np.array:
        x = x.astype("float32")
        x = x / 255.0
        x = x - 0.5
        x = x * 2.0
        return x

    @staticmethod
    def get_face_emotion_predictions(images: List[np.array]) -> List[Dict[str, float]]:
        channel = insecure_channel(f"{os.getenv(TFS_HOST_ENV)}:{os.getenv(TFS_PORT_ENV)}")
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        imgs_batch = np.empty((len(images), 64, 64, 1), dtype=np.float32)

        for idx, image_data in enumerate(images):
            gray_image = rgb2gray(image_data)
            gray_image = np.squeeze(gray_image)
            gray_image = gray_image.astype("uint8")
            gray_image = cv2.resize(gray_image, (64, 64))

            imgs_batch[idx] = np.expand_dims(gray_image, -1)

        imgs_batch = EmotionUtils.preprocess_input(imgs_batch)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = TF_SERVING_EMOTION_MODEL_NAME
        request.model_spec.signature_name = TF_SERVING_EMOTION_MODEL_SIGNATURE_NAME

        request.inputs[TF_SERVING_EMOTION_MODEL_INPUT].CopyFrom(make_tensor_proto(imgs_batch))

        try:
            response = stub.Predict(request, TFS_REQUESTS_TIMEOUT)
        except RpcError as e:
            handle_tfs_error(e, request.model_spec.name, logging.getLogger(FLASK_APP_NAME))
            raise TFSServiceUnavailableException()

        emotion_labels = EmotionUtils.get_labels(EMOTION_LABELS_DATASET)

        scores = np.array(response.outputs[TF_SERVING_EMOTION_MODEL_OUTPUT].float_val, dtype=np.float32).reshape(
            len(images), len(emotion_labels)
        )

        results = []

        for img_idx in range(len(scores)):
            result = dict()
            for jdx, score in enumerate(scores[img_idx]):
                result[emotion_labels[jdx]] = round(float(score), 2)
            results.append(result)

        return results
