import json

import binascii
import websocket
import websockets
from syft.serde.serde import serialize, deserialize

TIMEOUT_INTERVAL = 60


class GridClient:
    def __init__(self, id: str, address: str, secure: bool = False):
        self.id = id
        self.address = address
        self.secure = secure
        self.ws = None

    @property
    def url(self):
        return f"wss://{self.address}" if self.secure else f"ws://{self.address}"

    def connect(self):
        args = {"max_size": None, "timeout": TIMEOUT_INTERVAL, "url": self.url}

        self.ws = websocket.create_connection(**args)

    def _send_msg(self, message: dict) -> dict:
        """ Prepare/send a JSON message to a PyGrid server and receive the response.
            Args:
                message (dict) : message payload.
            Returns:
                response (dict) : response payload.
        """
        self.ws.send(json.dumps(message))
        return json.loads(self.ws.recv())

    def _serialize_object(self, obj):
        serialized_object = {}
        for k, v in obj.iteritems():
            serialized_object[k] = binascii.hexlify(serialize(v)).decode()
        return serialized_object

    def close(self):
        self.ws.shutdown()

    def host_federated_learning(
        self,
        model,
        client_plans,
        client_protocols,
        client_config,
        server_averaging_plan,
        server_config,
    ):
        serialized_model = binascii.hexlify(serialize(model)).decode()
        serialized_plans = self._serialize_object(client_plans)
        serialized_protocols = self._serialize_object(client_protocols)
        serialized_avg_plan = binascii.hexlify(serialize(server_averaging_plan)).decode()

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
