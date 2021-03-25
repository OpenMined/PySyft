# stdlib
import base64
import binascii
from collections import OrderedDict as PyOrderedDict
import json
import random
from timeit import timeit

# third party
import requests
import websocket

# syft relative
from .. import deserialize
from .. import serialize
from ..lib.python.collections.ordered_dict import OrderedDict
from ..lib.python.list import List
from ..proto.core.plan.plan_pb2 import Plan as PlanPB
from ..proto.lib.python.collections.ordered_dict_pb2 import OrderedDict as OrderedDictPB
from ..proto.lib.python.list_pb2 import List as ListPB
from .model_centric_fl_client import GridError

TIMEOUT_INTERVAL = 60
CHUNK_SIZE = 655360  # 640KB
SPEED_MULT_FACTOR = 10
MAX_BUFFER_SIZE = 1048576 * 64  # 64 MB
MAX_SPEED_TESTS = 3


class ModelCentricFLWorker:
    CYCLE_STATUS_ACCEPTED = "accepted"
    CYCLE_STATUS_REJECTED = "rejected"
    PLAN_TYPE_LIST = "list"
    PLAN_TYPE_TORCHSCRIPT = "torchscript"

    def __init__(self, id: str, address: str, secure: bool = False):
        self.id = id
        self.address = address
        self.secure = secure
        self.ws = None

    @property
    def ws_url(self):
        return f"wss://{self.address}" if self.secure else f"ws://{self.address}"

    @property
    def http_url(self):
        return f"https://{self.address}" if self.secure else f"http://{self.address}"

    def connect(self):
        args_ = {"max_size": None, "timeout": TIMEOUT_INTERVAL, "url": self.ws_url}

        self.ws = websocket.create_connection(**args_)

    def _send_msg(self, message: dict) -> dict:
        """Prepare/send a JSON message to a PyGrid server and receive the response.

        Args:
            message (dict) : message payload.
        Returns:
            response (dict) : response payload.
        """
        if self.ws is None or not self.ws.connected:
            self.connect()

        self.ws.send(json.dumps(message))
        json_response = json.loads(self.ws.recv())

        error = json_response["data"].get("error", None)
        if error is not None:
            raise GridError(error, None)

        return json_response

    def _send_http_req(
        self, method, path: str, params: dict = None, body: bytes = None
    ):
        if method == "GET":
            res = requests.get(self.http_url + path, params)
        elif method == "POST":
            res = requests.post(self.http_url + path, params=params, data=body)

        if not res.ok:
            raise GridError("HTTP response is not OK", res.status_code)

        response = res.content
        return response

    def _yield_chunk_from_request(self, request, chunk_size):
        for chunk in request.iter_content(chunk_size=chunk_size):
            yield chunk

    def _read_n_request_chunks(self, chunk_generator, n):
        for i in range(n):
            try:
                next(chunk_generator)
            except:
                return False
        return True

    def _get_ping(self, worker_id, random_id):
        params = {"is_ping": 1, "worker_id": worker_id, "random": random_id}
        ping = (
            timeit(
                lambda: self._send_http_req("GET", "/model-centric/speed-test", params),
                number=MAX_SPEED_TESTS,
            )
            * 1000
        )  # for ms
        return ping

    def _get_upload_speed(self, worker_id, random_id):
        buffer_size = CHUNK_SIZE
        speed_history = []

        for _ in range(MAX_SPEED_TESTS):
            data_sample = b"x" * buffer_size
            params = {"worker_id": worker_id, "random": random_id}
            body = {"upload_data": data_sample}
            time_taken = timeit(
                lambda: self._send_http_req(
                    "POST", "/model-centric/speed-test", params, body
                ),
                number=1,
            )
            if time_taken < 0.5:
                buffer_size = min(buffer_size * SPEED_MULT_FACTOR, MAX_BUFFER_SIZE)
            new_speed = buffer_size / (time_taken * 1024)

            if new_speed != float("inf"):
                speed_history.append(new_speed)

        if len(speed_history) == 0:
            return float("inf")
        else:
            avg_speed = sum(speed_history) / len(speed_history)
            return avg_speed

    def _get_download_speed(self, worker_id, random_id):
        params = {"worker_id": worker_id, "random": random_id}
        speed_history = []
        with requests.get(
            self.http_url + "/model-centric/speed-test", params, stream=True
        ) as r:
            r.raise_for_status()
            buffer_size = CHUNK_SIZE
            chunk_generator = self._yield_chunk_from_request(r, CHUNK_SIZE)
            for _ in range(MAX_SPEED_TESTS):
                time_taken = timeit(
                    lambda: self._read_n_request_chunks(
                        chunk_generator, buffer_size // CHUNK_SIZE
                    ),
                    number=1,
                )
                if time_taken < 0.5:
                    buffer_size = min(buffer_size * SPEED_MULT_FACTOR, MAX_BUFFER_SIZE)
                new_speed = buffer_size / (time_taken * 1024)

                if new_speed != float("inf"):
                    speed_history.append(new_speed)

        if len(speed_history) == 0:
            return float("inf")
        else:
            avg_speed = sum(speed_history) / len(speed_history)
            return avg_speed

    def _serialize(self, obj):
        """Serializes object to protobuf"""
        pb = serialize(obj)
        return pb.SerializeToString()

    def _serialize_object(self, obj):
        serialized_object = {}
        for k, v in obj.items():
            serialized_object[k] = binascii.hexlify(self._serialize(v)).decode()
        return serialized_object

    def _unserialize(self, serialized_obj, obj_protobuf_type):
        pb = obj_protobuf_type()
        pb.ParseFromString(serialized_obj)
        return deserialize(pb)

    def close(self):
        self.ws.shutdown()

    def authenticate(self, auth_token, model_name, model_version):
        message = {
            "type": "model-centric/authenticate",
            "data": {
                "auth_token": auth_token,
                "model_name": model_name,
                "model_version": model_version,
            },
        }

        return self._send_msg(message)

    def cycle_request(self, worker_id, model_name, model_version, speed_info):
        message = {
            "type": "model-centric/cycle-request",
            "data": {
                "worker_id": worker_id,
                "model": model_name,
                "version": model_version,
                **speed_info,
            },
        }
        return self._send_msg(message)

    def get_model(self, worker_id, request_key, model_id):
        params = {
            "worker_id": worker_id,
            "request_key": request_key,
            "model_id": model_id,
        }
        serialized_model = self._send_http_req(
            "GET", "/model-centric/get-model", params
        )
        return self._unserialize(serialized_model, ListPB)

    def get_plan(self, worker_id, request_key, plan_id, receive_operations_as):
        params = {
            "worker_id": worker_id,
            "request_key": request_key,
            "plan_id": plan_id,
            "receive_operations_as": receive_operations_as,
        }
        serialized_plan = self._send_http_req("GET", "/model-centric/get-plan", params)
        return self._unserialize(serialized_plan, PlanPB)

    def report(self, worker_id: str, request_key: str, diff: List):
        diff_serialized = self._serialize(List(diff))
        diff_base64 = base64.b64encode(diff_serialized).decode("ascii")
        params = {
            "type": "model-centric/report",
            "data": {
                "worker_id": worker_id,
                "request_key": request_key,
                "diff": diff_base64,
            },
        }
        res = self._send_msg(params)
        return res

    def get_connection_speed(self, worker_id):
        random_num = random.getrandbits(128)
        ping = self._get_ping(worker_id, random_num)
        upload_speed = self._get_upload_speed(worker_id, random_num)
        download_speed = self._get_download_speed(worker_id, random_num)
        return {"ping": ping, "download": download_speed, "upload": upload_speed}
