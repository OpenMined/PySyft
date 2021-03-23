import json
import binascii
import websocket
import requests

import syft as sy
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft.lib.python.collections import OrderedDict
from syft.lib.python import List
from syft.proto.lib.python.list_pb2 import List as ListPB
from syft import serialize, deserialize


TIMEOUT_INTERVAL = 60

class GridError(BaseException):
    def __init__(self, error, status):
        super().__init__(error)
        self.status = status
        self.error = error


class ModelCentricFLClient:
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

        # Look for error in root and under "data"
        error = None
        if "data" in json_response:
            error = json_response["data"].get("error", None)
        elif "error" in json_response:
            error = json_response["error"]
        if error is not None:
            raise GridError(error, None)

        return json_response

    def _send_http_req(self, method, path: str, params: dict = None, body: bytes = None):
        if method == "GET":
            res = requests.get(self.http_url + path, params)
        elif method == "POST":
            res = requests.post(self.http_url + path, params=params, data=body)

        if not res.ok:
            error = "HTTP response is not OK"
            try:
                json_response = json.loads(res.content)
                error = json_response.get("error", error)
            finally:
                raise GridError(f"Grid Error: {error}", res.status_code)

        response = res.content
        return response

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
        
    def hex_serialize(self, x):
        return binascii.hexlify(self._serialize(x)).decode()
        

    def host_federated_training(self, model, client_plans, client_protocols, client_config,
                                server_averaging_plan, server_config):
        
        serialized_model = self.hex_serialize(List(model.parameters()))
        serialized_plans = self._serialize_object(client_plans)
        serialized_protocols = self._serialize_object(client_protocols)
        serialized_avg_plan = self.hex_serialize(server_averaging_plan)

        # "model-centric/host-training" request body
        message = {
            "type": "model-centric/host-training",
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

    def get_model(self, name, version, checkpoint="latest"):
        params = {
            "name": name,
            "version": version,
            "checkpoint": checkpoint,
        }
        serialized_model = self._send_http_req("GET", "/model-centric/retrieve-model", params)
        return self._unserialize(serialized_model, ListPB)
