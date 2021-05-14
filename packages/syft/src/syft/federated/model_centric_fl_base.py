# stdlib
import binascii
import json
from typing import Any as TypeAny
from typing import Dict as TypeDict
from typing import Optional
from typing import Union as TypeUnion

# third party
from google.protobuf.message import Message
from google.protobuf.reflection import GeneratedProtocolMessageType
import requests
import websocket

# syft relative
from ..core.common.serde.deserialize import _deserialize as deserialize
from ..core.common.serde.serialize import _serialize as serialize
from ..federated import JSONDict

TIMEOUT_INTERVAL = 60


class GridError(BaseException):
    def __init__(
        self, error: TypeUnion[Exception, str], status: Optional[int] = None
    ) -> None:
        super().__init__(error)
        if type(error) is not str:
            error = str(error)
        self.error = error
        self.status = status


class ModelCentricFLBase:
    def __init__(self, address: str, secure: bool = False):
        self.address = address
        self.secure = secure
        self.ws: Optional[websocket._core.WebSocket] = None

    @property
    def ws_url(self) -> str:
        protocol = "wss" if self.secure else "ws"
        return f"{protocol}://{self.address}"

    @property
    def http_url(self) -> str:
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.address}"

    def connect(self) -> None:
        args_ = {"max_size": None, "timeout": TIMEOUT_INTERVAL, "url": self.ws_url}
        self.ws = websocket.create_connection(**args_)

    def _send_msg(self, message: TypeDict[str, TypeAny]) -> JSONDict:
        """Prepare/send a JSON message to a PyGrid server and receive the response.

        Args:
            message (dict) : message payload.
        Returns:
            response (dict) : response payload.
        """
        if self.ws is None or not self.ws.connected:
            self.connect()

        if self.ws is not None:
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
        else:
            raise GridError("Websocket connection unavailable", None)

    def _send_http_req(
        self,
        method: str,
        path: str,
        params: Optional[JSONDict] = None,
        body: Optional[JSONDict] = None,
    ) -> bytes:
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

        return res.content

    def _serialize(self, obj: object) -> bytes:
        """Serializes object to protobuf"""
        pb: Message = serialize(obj)  # type: ignore
        return pb.SerializeToString()

    def _serialize_dict_values(self, obj: JSONDict) -> JSONDict:
        serialized_object = {}
        for k, v in obj.items():
            serialized_object[k] = binascii.hexlify(self._serialize(v)).decode()
        return serialized_object

    def _unserialize(
        self, serialized_obj: bytes, obj_protobuf_type: GeneratedProtocolMessageType
    ) -> TypeAny:
        pb = obj_protobuf_type()
        pb.ParseFromString(serialized_obj)
        return deserialize(pb)

    def close(self) -> None:
        if self.ws is not None:
            self.ws.shutdown()

    def hex_serialize(self, x: object) -> str:
        return binascii.hexlify(self._serialize(x)).decode()
