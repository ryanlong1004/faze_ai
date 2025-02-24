from typing import Any, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from faces_api_utils import singleton

import logic.facenet_detect.detect_face as mtcnn
from config.settings import FACE_DETECTION_THRESHOLD


@singleton
class FaceDetection(object):
    def __init__(self) -> None:
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

            with sess.as_default():
                pnet, rnet, onet = mtcnn.create_mtcnn(sess, None)

        self._model = {"pnet": pnet, "rnet": rnet, "onet": onet}

    def get_model(self):
        return self._model


def detect_on_single_image(
    image: np.array, return_aligned_images: bool = False, output_image_size: int = 160
) -> Tuple[np.array, Iterator[Union[float, Any]], Optional[List[np.array]]]:
    face_detection_model = FaceDetection().get_model()

    pnet = face_detection_model["pnet"]
    rnet = face_detection_model["rnet"]
    onet = face_detection_model["onet"]

    minsize = 20  # minimum size of face
    # three steps's threshold - [0.6, 0.7, 0.7]
    threshold = FACE_DETECTION_THRESHOLD
    factor = 0.709  # scale factor
    margin = 30

    ret_bboxes = []
    ret_images = []

    bounding_boxes, _ = mtcnn.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]

    # get the confidence values separate, and get only two digits precision
    confidence_values = map(lambda x: round(x, 4), bounding_boxes[:, -1])

    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(image.shape)[0:2]
        if nrof_faces > 1:
            for i in range(nrof_faces):
                det_arr.append(np.squeeze(det[i]))
        else:
            det_arr.append(np.squeeze(det))

        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)

            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)

            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])

            if return_aligned_images:
                cropped = image[bb[1] : bb[3], bb[0] : bb[2], :]
                scaled = cv2.resize(cropped, (output_image_size, output_image_size), interpolation=cv2.INTER_LINEAR)
                ret_images.append(scaled)

            ret_bboxes.append([bb[0], bb[1], bb[2], bb[3]])

    if return_aligned_images:
        return np.array(ret_bboxes), confidence_values, ret_images
    else:
        return np.array(ret_bboxes), confidence_values, None
