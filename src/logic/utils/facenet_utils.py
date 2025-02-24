from __future__ import absolute_import, division, print_function

import logging
import math
import os
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from grpc import RpcError, _channel, insecure_channel
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from config.faces_api_exceptions import TFSServiceUnavailableException
from config.settings import (DEFAULT_FACE_IMAGE_SIZE, EMBEDDING_SIZE,
                             FLASK_APP_NAME, TFS_EMBEDDINGS_OPERATION,
                             TFS_HOST_ENV, TFS_MODEL_NAME, TFS_PORT_ENV,
                             TFS_REQUESTS_TIMEOUT)
from logic.utils.tf_utils import handle_tfs_error


class FacenetUtils:
    @staticmethod
    def get_channel_and_stub() -> Tuple[_channel.Channel, prediction_service_pb2_grpc.PredictionServiceStub]:
        channel = insecure_channel(f"{os.getenv(TFS_HOST_ENV)}:{os.getenv(TFS_PORT_ENV)}")
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        return channel, stub

    @staticmethod
    def prewhiten(x: np.array) -> np.array:
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def get_embeddings_from_images(
        self, images_list: list, batch_size: int = 100, image_size: int = DEFAULT_FACE_IMAGE_SIZE
    ) -> np.ndarray:
        nrof_samples = len(images_list)
        images = np.zeros((nrof_samples, image_size, image_size, 3), dtype=np.float32)

        for i, img in enumerate(images_list):
            img = FacenetUtils.prewhiten(img)

            # check correct size
            if img.shape[:2] != (image_size, image_size):
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            images[i] = img

        channel, stub = self.get_channel_and_stub()

        request = predict_pb2.PredictRequest()
        request.model_spec.name = TFS_MODEL_NAME
        request.model_spec.signature_name = TFS_EMBEDDINGS_OPERATION

        # Run forward pass to calculate embeddings
        nrof_batches_per_epoch = int(math.ceil(nrof_samples / batch_size))
        emb_array = np.zeros((nrof_samples, EMBEDDING_SIZE))

        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_samples)

            images_batch = images[start_index:end_index]

            request.inputs["images"].CopyFrom(
                make_tensor_proto(images_batch, shape=images_batch.shape, dtype=tf.float32)
            )

            request.inputs["phase"].CopyFrom(tf.contrib.util.make_tensor_proto(False))

            try:
                result = stub.Predict(request, TFS_REQUESTS_TIMEOUT)
            except RpcError as e:
                handle_tfs_error(e, request.model_spec.name, logging.getLogger(FLASK_APP_NAME))
                raise TFSServiceUnavailableException()

            np_res = np.array(result.outputs["embeddings"].float_val).reshape((len(images_batch), EMBEDDING_SIZE))
            emb_array[start_index:end_index] = np_res

        return emb_array
