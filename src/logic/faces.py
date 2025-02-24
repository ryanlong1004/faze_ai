import base64
import binascii
from io import BytesIO
from typing import Any, Dict, List, Tuple, Union

import imageio
import numpy as np
from data_access.data_access import DataAccess
from scipy.spatial import cKDTree
from skimage import transform
from sklearn.cluster import AgglomerativeClustering

import logic.utils.emotion_utils
import logic.utils.gender_utils
from config.faces_api_exceptions import (
    BadBatchLengthForGroupingRecognition,
    ClientExistsException,
    ClientNotExistsException,
    InvalidBase64CodeException,
    PersonNotExistsException,
)
from config.settings import (
    DEFAULT_CELEBRITY_RECOGNITION_COEFFICIENT,
    DEFAULT_CELEBRITY_RECOGNITION_THRESHOLD,
    DEFAULT_CLUSTERING_RECOGNITION_THRESHOLD,
    DEFAULT_FACE_IMAGE_SIZE,
    DEFAULT_RECOGNITION_THRESHOLD,
    LOWER_RECOGNITION_THRESHOLD,
    UNKNOWN_FACES_LIMIT,
)
from logic.utils.celebs_utils import CelebsUtils
from logic.utils.detection_utils import detect_on_single_image
from logic.utils.facenet_utils import FacenetUtils
from logic.utils.noise_utils import get_noise_predictions


def recognition_confidence_from_distance(distance: float) -> float:
    confidence: float = round(1 - (np.arctan(distance / 1.1) / (np.pi / 2)), 2)
    return confidence


def scale_images(images: List[np.ndarray]) -> List[np.ndarray]:
    # ensure that all images are equally scaled
    scaled_images = []
    for image in images:
        scaled_images.append(
            transform.resize(
                image, (DEFAULT_FACE_IMAGE_SIZE, DEFAULT_FACE_IMAGE_SIZE), mode="constant", anti_aliasing=True
            )
        )
    return scaled_images


class FacesLogic:
    def __init__(self, data_access: DataAccess, facenet_utils: FacenetUtils, celebs_utils: CelebsUtils):
        self.data_access = data_access
        self.facenet = facenet_utils
        self.celebs = celebs_utils

    def detect_align_faces(self, client_id: str, image: np.ndarray) -> Any:
        if not self.data_access.client_exists(client_id):
            raise ClientNotExistsException()

        try:
            image_bytes = base64.b64decode(image)
            image = imageio.imread(BytesIO(image_bytes), pilmode="RGB")
        except (TypeError, binascii.Error, OSError, ValueError) as exc:
            raise InvalidBase64CodeException() from exc
        return detect_on_single_image(image, True)

    def get_noise_predictions(self, client_id: str, image: np.ndarray) -> float:
        if not self.data_access.client_exists(client_id):
            raise ClientNotExistsException()
        return get_noise_predictions(image)

    def get_faces_emotions(self, client_id: str, images: List[np.ndarray]) -> List[Dict[str, float]]:
        if not self.data_access.client_exists(client_id):
            raise ClientNotExistsException()
        return logic.utils.emotion_utils.EmotionUtils().get_face_emotion_predictions(images)

    def get_faces_gender(self, client_id: str, images: List[np.ndarray]) -> List[Dict[str, float]]:
        if not self.data_access.client_exists(client_id):
            raise ClientNotExistsException()
        return logic.utils.gender_utils.get_face_gender_predictions(images)

    def create_client(self, client_id: str, base_dataset: str = "empty") -> Dict[str, Union[int, str, list]]:
        if self.data_access.client_exists(client_id):
            raise ClientExistsException()
        return self.data_access.create_client(client_id, base_dataset)

    def client_exists(self, client_id: str) -> bool:
        return self.data_access.client_exists(client_id)

    def delete_client(self, client_id: str) -> bool:
        if not self.data_access.client_exists(client_id):
            raise ClientNotExistsException()
        return self.data_access.delete_client(client_id)

    def get_embeddings(self, faces_images: List[np.ndarray]) -> np.ndarray:
        scaled_images = scale_images(faces_images)
        embs = self.facenet.get_embeddings_from_images(scaled_images, image_size=DEFAULT_FACE_IMAGE_SIZE)
        return embs

    @staticmethod
    def generate_clusters(embeddings: np.ndarray) -> np.ndarray:
        """
        Applies a clustering algorithm to group similar embeddings, labels from those clusters are returned as results.
        :param embeddings: list of embeddings to group.
        :return: labels of the clusters for the given embeddings.
        """

        # More info about the clustering algorithm and its parameters can be found at
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
        clustering = AgglomerativeClustering(
            affinity="euclidean",
            linkage="average",
            distance_threshold=DEFAULT_CLUSTERING_RECOGNITION_THRESHOLD,
            n_clusters=None,
        ).fit(embeddings)
        return clustering.labels_

    @staticmethod
    def get_representative_sample(vectors: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Searches for the centroid of list of points (represented by their embeddings) and the closest point to it.
        :param vectors: list of embeddings for all the points in the cluster.
        :return: A tuple with the index of the nearest point to the centroid, the nearest embedding and the
        found centroid.
        """
        centroid = np.mean(vectors, axis=0)
        nearest_idx = cKDTree(vectors).query(centroid, k=1)[1]
        nearest = vectors[nearest_idx]
        return nearest_idx, nearest, centroid

    def __get_nearest_face(
        self,
        client_id: str,
        embedding: np.ndarray,
        image_id: str,
        recognition_threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
        only_unknowns: bool = False,
    ) -> Tuple[str, str, float, bool]:
        """
        Searches for a matching face on the database for the given embedding in the specified recognition threshold.
        :param client_id: client's id.
        :param embedding: vector representation of the face.
        :param image_id: image's id.
        :param recognition_threshold: threshold used as distance to recognize a face as a saved person or not.
        :param only_unknowns: to search only on unknowns.
        :return: A tuple with the query results of person id, person name, confidence and if there was a match.
        """
        nearest_face_distance = recognition_threshold + 1  # initialize to a non match value
        if not only_unknowns:
            try:
                nearest_face = self.data_access.get_nearest_faces(
                    client_id, embedding, threshold=recognition_threshold
                )[0]
                person_id = nearest_face["person_id"]
                nearest_face_distance = float(nearest_face["distance"])
            except IndexError:
                # there was no result, no known faces in the dataset yet
                pass
        if nearest_face_distance > recognition_threshold:
            known_face = False
            try:
                uf_nearest_face = self.data_access.get_nearest_faces(
                    client_id, embedding, search_unknowns=True, threshold=recognition_threshold
                )[0]
                uf_distance: float = uf_nearest_face["distance"]
                uf_person_id = uf_nearest_face["person_id"]
            except IndexError:
                # there was no result, no unknown in the dataset yet
                uf_distance = recognition_threshold + 1
            confidence = recognition_confidence_from_distance(uf_distance)
            if uf_distance < recognition_threshold:
                person_id = uf_person_id
                # if it's a match, but it's distance is more than LOWER_RECOGNITION_THRESHOLD, save it
                if uf_distance >= LOWER_RECOGNITION_THRESHOLD:
                    # limit the number of faces for unknowns
                    faces_count = len(self.data_access.get_person_faces_ids(person_id)["face_ids"])
                    if faces_count < UNKNOWN_FACES_LIMIT:
                        self.data_access.insert_face(image_id, person_id, client_id, embedding)
            else:
                person_id = self.data_access.get_new_person_id()
                # confidence that this is a new face is the complement of the confidence for a known face
                confidence = 1 - confidence
                # learn the new face
                self.data_access.insert_person(client_id, person_id, "unknown", False)
                self.data_access.insert_face(image_id, person_id, client_id, embedding)
        else:
            known_face = True
            confidence = recognition_confidence_from_distance(nearest_face_distance)

        person_name: str = nearest_face["name"] if known_face else "unknown"

        # person_id, person_name, confidence, matched
        return str(person_id), person_name, round(confidence, 2), known_face

    def get_nearest_faces_by_grouping(
        self,
        client_id: str,
        faces_images_ids: List[str],
        embeddings: np.ndarray,
        recognition_threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
        only_unknowns: bool = False,
    ) -> List[Dict[str, Union[str, float, List[str]]]]:
        """
        Given a batch of faces' embeddings it applies a clustering algorithm to find similar groups of faces. Then
        the centroid is calculated for each cluster and the nearest point to that centroid. This nearest point is
        sent for recognition and all the faces belonging to the cluster are labeled in the same way.
        IMPORTANT!: the batch's length CANNOT be less than 3, since it is a clustering algorithm's requirement.
        If batch's length is less than 3 an exception of BadBatchLengthForGroupingRecognition will be raised.
        :param client_id: client's id
        :param faces_images_ids: list of ids for the faces images
        :param embeddings: list of image's embeddings.
        :param recognition_threshold: threshold used as distance to recognize a face as a saved person or not
        :param only_unknowns: to search only on unknowns
        :return: a list with the search results for the given set of embeddings.
        """

        if len(embeddings) < 3:
            raise BadBatchLengthForGroupingRecognition()

        # generate clusters
        labels = FacesLogic.generate_clusters(embeddings)
        clusters = np.unique(labels)

        results: List[Dict[str, Union[str, float, List[str]]]] = [{}] * len(embeddings)
        for cluster in clusters:
            filter_idxs = np.where(labels == cluster)[0]
            nearest_idx, nearest, _ = FacesLogic.get_representative_sample(list(embeddings[filter_idxs]))
            true_nearest_idx = filter_idxs[nearest_idx]

            emb = nearest  # embs[true_nearest_idx]
            image_id = faces_images_ids[true_nearest_idx]
            person_id, person_name, confidence, matched = self.__get_nearest_face(
                client_id, emb, image_id, recognition_threshold, only_unknowns
            )
            for idx in filter_idxs:
                results[idx] = {
                    "person_id": str(person_id),
                    "person_name": person_name,
                    "recognition_confidence": confidence,
                    "face_id": faces_images_ids[idx],
                    "matched": matched,
                }

        return results

    def get_nearest_faces_by_one_on_one(
        self,
        client_id: str,
        faces_images_ids: List[str],
        embeddings: np.ndarray,
        recognition_threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
        only_unknowns: bool = False,
    ) -> List[Dict[str, Union[str, float, bool]]]:
        """
        Given a batch of faces' embeddings it sends them one by one for recognition and returns back the results of it.
        :param client_id: client's id.
        :param faces_images_ids: list of ids for the faces images.
        :param embeddings: list of image's embeddings.
        :param recognition_threshold: threshold used as distance to recognize a face as a saved person or not
        :param only_unknowns: to search only on unknowns.
        :return: a list with the search results for the given set of embeddings.
        """
        results = []
        for emb, image_id in zip(embeddings, faces_images_ids):
            person_id, person_name, confidence, matched = self.__get_nearest_face(
                client_id, emb, image_id, recognition_threshold, only_unknowns
            )
            results.append(
                {
                    "person_id": str(person_id),
                    "person_name": person_name,
                    "recognition_confidence": confidence,
                    "face_id": image_id,
                    "matched": matched,
                }
            )
        return results

    def get_nearest_faces(
        self,
        client_id: str,
        faces_images: List[np.ndarray],
        faces_images_ids: List[str],
        recognition_threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
        only_unknowns: bool = False,
        grouping: bool = False,
    ) -> List[dict]:
        """
        Sends a list of face images for recognition, depending on grouping parameter it will send all of them as a
        batch (when True) or one by one (when False). If grouping is True, then the the batch's length must be greater
        or equal to 3.
        :param client_id: client's id.
        :param faces_images: list of image's (in a n-array form).
        :param faces_images_ids: list of ids for the faces images.
        :param recognition_threshold: threshold used as distance to recognize a face as a saved person or not
        :param only_unknowns: to search only on unknowns.
        :param grouping: to apply (or not) grouping approach for the given faces before recognition
        operation.
        :return: a list with the search results for the given set of embeddings.
        """
        if not self.data_access.client_exists(client_id):
            raise ClientNotExistsException()

        scaled_images = scale_images(faces_images)
        embeddings = self.facenet.get_embeddings_from_images(scaled_images, image_size=DEFAULT_FACE_IMAGE_SIZE)

        if grouping:
            results = self.get_nearest_faces_by_grouping(
                client_id, faces_images_ids, embeddings, recognition_threshold, only_unknowns
            )
        else:
            results = self.get_nearest_faces_by_one_on_one(
                client_id, faces_images_ids, embeddings, recognition_threshold, only_unknowns
            )
        return results

    def learn_images_for_known_person(
        self, client_id: str, person_id: str, person_name: str, images: List[np.ndarray], images_ids: List[str]
    ) -> List[dict]:
        return self.learn_images(client_id, person_id, person_name, images, images_ids, True)

    def learn_images(
        self,
        client_id: str,
        person_id: str,
        person_name: str,
        images: List[np.ndarray],
        images_ids: List[str],
        known: bool,
    ) -> List[Dict[str, str]]:
        if not self.data_access.client_exists(client_id):
            raise ClientNotExistsException()

        person_id = str(person_id)
        # ensure that all images are equally scaled
        scaled_images = scale_images(images)
        new_embs = self.facenet.get_embeddings_from_images(scaled_images, image_size=DEFAULT_FACE_IMAGE_SIZE)

        # check whether this is person or a new set of images for an existing person
        if not self.data_access.person_exists(person_id):
            # new person!
            self.data_access.insert_person(client_id, person_id, person_name, known)

        # keep track of replaced faces (when learning faces that already exist in the faces db)
        replacements = []

        for image_id, new_emb in zip(images_ids, new_embs):
            try:
                # check whether this is a name assignment for an existing face (known or unknown)
                existing_face = self.data_access.get_nearest_faces(
                    client_id, new_emb, search_unknowns=None, exact_only=True
                )[0]

                existing_person = self.data_access.get_person(existing_face["person_id"])

                # remove existing face
                self.data_access.delete_faces(client_id, [existing_face["id"]])

                # keep track of replaced faces
                replacements.append(
                    {
                        "face_id": image_id,
                        "old_face_id": existing_face["id"],
                        "old_person_id": existing_face["person_id"],
                        "old_person_type": "known" if existing_person["known"] else "unknown",
                    }
                )
            except IndexError:
                # there was no result, the face is not already in the db
                pass
            # insert face with new face_id and person_id
            self.data_access.insert_face(image_id, person_id, client_id, new_emb)

        # return replaced faces info
        return replacements

    def get_nearest_celebrity_face(
        self, face_image: np.ndarray, recognition_threshold: float = DEFAULT_CELEBRITY_RECOGNITION_THRESHOLD
    ) -> Dict[str, Union[float, str]]:
        scaled_image = scale_images([face_image])[0]

        emb = self.facenet.get_embeddings_from_images([scaled_image], image_size=DEFAULT_FACE_IMAGE_SIZE)[0]

        try:
            nearest_face = self.celebs.get_nearest_celeb(emb)
            nearest_face_distance = float(nearest_face["distance"])
        except IndexError:
            # there was no result, no known faces in the dataset yet
            nearest_face_distance = recognition_threshold + 1
        if nearest_face_distance > recognition_threshold:
            result = {}
        else:
            confidence = recognition_confidence_from_distance(
                nearest_face_distance * DEFAULT_CELEBRITY_RECOGNITION_COEFFICIENT
            )
            result = {
                "celebrity_name": nearest_face["name"],
                "celebrity_id": nearest_face["celeb_id"],
                "confidence": confidence,
            }
        return result

    def get_person_faces_ids(self, client_id: str, person_id: str) -> Dict[str, List[str]]:
        if not self.data_access.client_exists(client_id):
            raise ClientNotExistsException()
        return self.data_access.get_person_faces_ids(person_id)

    def get_new_person_id(self, client_id: str) -> str:
        if not self.data_access.client_exists(client_id):
            raise ClientNotExistsException()
        return str(self.data_access.get_new_person_id())

    def update_person(self, client_id: str, person_id: str, name: str, known: bool) -> Dict[str, Union[str, int, bool]]:
        if not self.data_access.client_exists(client_id):
            raise ClientNotExistsException()
        return self.data_access.update_person(client_id, person_id, name, known)

    def delete_person(self, person_id: str) -> bool:
        if not self.data_access.person_exists(person_id):
            raise PersonNotExistsException()
        return self.data_access.delete_person(person_id)

    def delete_faces(self, client_id: str, faces_ids: List[str]) -> Dict[str, bool]:
        if not self.data_access.client_exists(client_id):
            raise ClientNotExistsException()
        return self.data_access.delete_faces(client_id, faces_ids)
