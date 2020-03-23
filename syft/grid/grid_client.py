import binascii
import websocket
import websockets

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
        encoded_model = str(binascii.hexlify(model))

        message = {
            "type": "host_federated_learning",
            "model": model,
            "plans": client_plans,
            "protocols": client_protocols,
            "client_config": client_config,
            "server_config": server_config,
            "averaging_plan": server_averaging_plan
        }
