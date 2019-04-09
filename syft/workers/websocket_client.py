import binascii
from typing import Union
from typing import List

import torch
import websocket
import time
import logging

import syft as sy
from syft.codes import MSGTYPE
from syft.frameworks.torch.tensors.interpreters import AbstractTensor
from syft.workers import BaseWorker

logger = logging.getLogger(__name__)


class WebsocketClientWorker(BaseWorker):
    def __init__(
        self,
        hook,
        host: str,
        port: int,
        id: Union[int, str] = 0,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor, AbstractTensor]] = None,
    ):
        """A client which will forward all messages to a remote worker running a
        WebsocketServerWorker and receive all responses back from the server.
        """

        # TODO get angry when we have no connection params
        self.port = port
        self.host = host
        self.uri = f"ws://{self.host}:{self.port}"

        # creates the connection with the server which gets held open until the
        # WebsocketClientWorker is garbage collected.
        self.ws = websocket.create_connection(self.uri, max_size=None)

        super().__init__(hook, id, data, is_client_worker, log_msgs, verbose)

    def search(self, *query):
        # Prepare a message requesting the websocket server to search among its objects
        message = (MSGTYPE.SEARCH, query)
        serialized_message = sy.serde.serialize(message)
        # Send the message and return the deserialized response.
        response = self._recv_msg(serialized_message)
        return sy.serde.deserialize(response)

    def _send_msg(self, message: bin, location) -> bin:
        raise RuntimeError(
            "_send_msg should never get called on a ",
            "WebsocketClientWorker. Did you accidentally "
            "make hook.local_worker a WebsocketClientWorker?",
        )

    def _receive_action(self, message: bin) -> bin:
        self.ws.send(str(binascii.hexlify(message)))
        response = binascii.unhexlify(self.ws.recv()[2:-1])
        return response

    def _recv_msg(self, message: bin) -> bin:
        """Forwards a message to the WebsocketServerWorker"""

        response = self._receive_action(message)
        if not self.ws.connected:
            logger.warning("Websocket connection closed (worker: %s)", self.id)
            self.ws.shutdown()
            time.sleep(1)
            self.ws = websocket.create_connection(self.uri, max_size=None)
            logger.warning("Created new websocket connection")
            time.sleep(0.1)
            response = self._receive_action(message)
            if not self.ws.connected:
                raise RuntimeError(
                    "Websocket connection closed and creation of new connection failed."
                )
        return response
