import json

import binascii
import base64
import websocket
import requests

import syft as sy
from syft.serde import protobuf

from syft.execution.state import State
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft_proto.execution.v1.protocol_pb2 import Protocol as ProtocolPB

TIMEOUT_INTERVAL = 60


class GridError(BaseException):
    def __init__(self, error, status):
        self.status = status
        self.error = error


class GridClient:
    CYCLE_STATUS_ACCEPTED = "accepted"
    CYCLE_STATUS_REJECTED = "rejected"
    PLAN_TYPE_LIST = "list"
    PLAN_TYPE_TORCHSCRIPT = "torchscript"

    def __init__(self, id: str, address: str, secure: bool = False):
        self.id = id
        self.address = address
        self.secure = secure
        self.ws = None
        self.serialize_worker = sy.VirtualWorker(hook=None)

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
        """ Prepare/send a JSON message to a PyGrid server and receive the response.

        Args:
            message (dict) : message payload.
        Returns:
            response (dict) : response payload.
        """
        if self.ws is None or not self.ws.connected:
            self.connect()

        self.ws.send(json.dumps(message))
        json_response = json.loads(self.ws.recv())

        # print("REQ", message)
        # print("RES", json_response)

        error = json_response["data"].get("error", None)
        if error is not None:
            raise GridError(error, None)

        return json_response

    def _send_http_req(self, method, path: str, params: dict = None, body: bytes = None):
        if method == "GET":
            res = requests.get(self.http_url + path, params)
        elif method == "POST":
            res = requests.post(self.http_url + path, body)

        if not res.ok:
            raise GridError("HTTP response is not OK", res.status_code)

        response = res.content
        return response

    def _serialize(self, obj):
        """Serializes object to protobuf"""
        pb = protobuf.serde._bufferize(self.serialize_worker, obj)
        return pb.SerializeToString()

    def _serialize_object(self, obj):
        serialized_object = {}
        for k, v in obj.items():
            serialized_object[k] = binascii.hexlify(self._serialize(v)).decode()
        return serialized_object

    def _unserialize(self, serialized_obj, obj_protobuf_type):
        pb = obj_protobuf_type()
        pb.ParseFromString(serialized_obj)
        serialization_worker = sy.VirtualWorker(hook=None, auto_add=False)
        return protobuf.serde._unbufferize(serialization_worker, pb)

    def close(self):
        self.ws.shutdown()

    def host_federated_training(
        self,
        model,
        client_plans,
        client_protocols,
        client_config,
        server_averaging_plan,
        server_config,
    ):
        serialized_model = binascii.hexlify(self._serialize(model)).decode()
        serialized_plans = self._serialize_object(client_plans)
        serialized_protocols = self._serialize_object(client_protocols)
        serialized_avg_plan = binascii.hexlify(self._serialize(server_averaging_plan)).decode()

        # "federated/host-training" request body
        message = {
            "type": "federated/host-training",
            "data": {
                "model": serialized_model,
                "plans": serialized_plans,
                "protocols": serialized_protocols,
                "averaging_plan": serialized_avg_plan,
                "client_config": client_config,
                "server_config": server_config,
            },
        }

        return self._send_msg(message)

    def authenticate(self, auth_token):
        message = {
            "type": "federated/authenticate",
            "data": {"auth_token": auth_token},
        }

        return self._send_msg(message)

    def cycle_request(self, worker_id, model_name, model_version, speed_info):
        message = {
            "type": "federated/cycle-request",
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
        serialized_model = self._send_http_req("GET", "/federated/get-model", params)
        return self._unserialize(serialized_model, StatePB)

    def get_plan(self, worker_id, request_key, plan_id, receive_operations_as):
        params = {
            "worker_id": worker_id,
            "request_key": request_key,
            "plan_id": plan_id,
            "receive_operations_as": receive_operations_as,
        }
        serialized_plan = self._send_http_req("GET", "/federated/get-plan", params)
        return self._unserialize(serialized_plan, PlanPB)

    def get_protocol(self, worker_id, request_key, protocol_id):
        params = {
            "worker_id": worker_id,
            "request_key": request_key,
            "plan_id": protocol_id,
        }
        serialized_protocol = self._send_http_req("GET", "/federated/get-protocol", params)
        return self._unserialize(serialized_protocol, ProtocolPB)

    def report(self, worker_id: str, request_key: str, diff: State):
        diff_serialized = self._serialize(diff)
        diff_base64 = base64.b64encode(diff_serialized).decode("ascii")
        params = {
            "type": "federated/report",
            "data": {"worker_id": worker_id, "request_key": request_key, "diff": diff_base64},
        }
        return self._send_msg(params)

    def get_connection_speed(self, worker_id):
        # TODO
        return {"ping": 5, "download": 100, "upload": 100}
