import socket
import struct
import typing

import joblib
from numpy import ndarray

from config.settings import (CELEBRITES_MODEL_HOST, CELEBRITES_MODEL_PORT,
                             CELEBRITES_NAMES_PATH)

# Constants
# Keep alive message for the internal SPTAG protocol.
CELEBS_SPTAG_KEEP_ALIVE = b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
# Hello message for the internal SPTAG protocol.
CELEBS_SPTAG_CONNECT = b"\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
# Numerical Constants for internal use in the SPTAG protocol.
_256_256 = 256 * 256
_256_256_256 = 256 * 256 * 256

with open(CELEBRITES_NAMES_PATH, "rb") as f:
    CELEBS_NAMES_IDS: typing.List[typing.Tuple[str, str, str]] = joblib.load(f)

CELEBS_ID_NAME_MAP: typing.Dict[str, str] = {}
for celeb_id, _, celeb_name in CELEBS_NAMES_IDS:
    CELEBS_ID_NAME_MAP[celeb_id] = celeb_name


class _SPTAGClient:
    def __init__(self, host: str = CELEBRITES_MODEL_HOST, port: int = CELEBRITES_MODEL_PORT) -> None:
        self.host = host
        self.port = port

    @staticmethod
    def _int_to_bytes(i: int) -> bytearray:
        return bytearray([i % 256, (i // 256) % 256, (i // _256_256) % 256, (i // _256_256_256) % 256])

    @staticmethod
    def _bytes_to_int(b_arr: bytes) -> int:
        return b_arr[0] + 256 * b_arr[1] + _256_256 * b_arr[2] + _256_256_256 * b_arr[3]

    @staticmethod
    def _build_query_header_message(raw_query: str) -> typing.Tuple[bytearray, bytearray]:
        utf8_qry = bytearray(raw_query, encoding="UTF-8")

        header = bytearray([3, 0])
        header.extend(_SPTAGClient._int_to_bytes(len(utf8_qry) + 9))
        header.extend(bytearray(4))
        header.extend(_SPTAGClient._int_to_bytes(1))
        header.extend(bytearray(2))

        message = bytearray([1, 0, 0, 0, 0])
        message.extend(_SPTAGClient._int_to_bytes(len(utf8_qry)))
        message.extend(utf8_qry)

        return header, message

    @staticmethod
    def _build_query_str(_vector: ndarray) -> str:
        query = "query\t{qry}|".format(qry="|".join([str(x) for x in _vector]))
        return query

    def send_query(self, _vector: ndarray) -> typing.List[typing.Tuple[int, float]]:
        query = _SPTAGClient._build_query_str(_vector)

        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the port where the server is listening
        server_address = self.host, self.port
        sock.connect(server_address)

        try:
            header, message = _SPTAGClient._build_query_header_message(query)

            # Connect
            sock.sendall(CELEBS_SPTAG_CONNECT)
            sock.sendall(CELEBS_SPTAG_KEEP_ALIVE)

            # Send message
            sock.sendall(header)
            sock.sendall(message)

            # discard headers
            for _ in range(5):
                sock.recv(16)

            data = sock.recv(5)
            records_count = _SPTAGClient._bytes_to_int(data)

            results = []

            for _ in range(records_count):
                data = sock.recv(8)

                vector_index = _SPTAGClient._bytes_to_int(data[:4])
                vector_value = struct.unpack("f", data[4:])[0]

                results.append((vector_index, vector_value))  # Both pairs of parenthesis are needed here

            return results

        finally:
            if sock:
                sock.close()


class CelebsUtils:
    @staticmethod
    def vote_value(values: typing.List[typing.Tuple[str, float]]) -> typing.Tuple[str, float]:
        """
            If the input list has only one pair, the function returns it.
            If not; it groups all pairs by the celeb_id:
                - If there's only one group; it returns the first pair of it
                - If there's a group with more number of pairs than every other group;
                    it returns the first pair of this group.
                - If there's more than one group with the most number of pairs; from these groups,
                    it takes the one with the lowest mean distance and returns the first pair of that group.
        :param values:
            List of pairs with the celeb_id and the distance to the query vector, assumed order
            from least distant to most distant.
        :return:
            A pair with the celeb_id and distance to query vector that voted as the more suitable in the
                k-nearest approach.
        """

        k = len(values)

        if k == 1:
            celeb_id, celeb_dist = values[0]
        else:
            max_len = 0
            id_map: typing.Dict[str, int] = {}
            dist_map: typing.Dict[str, typing.List[float]] = {}
            for id_x, dist in values:
                new_len = id_map.get(id_x, 0) + 1
                id_map[id_x] = new_len
                if new_len > max_len:
                    max_len = new_len

                # mapping of lengths
                if id_x in dist_map:
                    dist_map[id_x].append(dist)
                else:
                    dist_map[id_x] = [dist]

            maximals = [id_x for id_x in id_map if id_map[id_x] == max_len]
            if len(maximals) == 1:
                # there is a group with most faces near the query
                celeb_id = maximals[0]
                celeb_dist = dist_map[celeb_id][0]
            else:
                # there are various groups with the SAME QUANTITY of faces near the query
                celeb_id = maximals[0]
                min_dist = sum(dist_map[celeb_id])
                for id_x in maximals[1:]:
                    # compute the average for each group, and then get the one closer
                    sum_dist = sum(dist_map[id_x])
                    if sum_dist < min_dist:
                        min_dist = sum_dist
                        celeb_id = id_x
                celeb_dist = dist_map[celeb_id][0]  # first value of distance of the celeb_id identified-
        return celeb_id, celeb_dist

    @staticmethod
    def get_nearest_celeb(face_emb: ndarray) -> typing.Dict[str, typing.Any]:
        _client = _SPTAGClient()

        value_pairs = _client.send_query(_vector=face_emb)
        name_value_pairs = [(CELEBS_NAMES_IDS[idx][0], value) for idx, value in value_pairs]
        celeb_id, val = CelebsUtils.vote_value(name_value_pairs)

        # get celebrity name from id.
        celeb_name = CELEBS_ID_NAME_MAP[celeb_id]

        return {"name": celeb_name, "celeb_id": celeb_id, "distance": round(abs(val), 2)}
