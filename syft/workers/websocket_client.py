import binascii
from typing import Union
from typing import List

import torch
import websocket
import time
import logging
import ssl

import syft as sy
from syft.codes import MSGTYPE
from syft.frameworks.torch.tensors.interpreters import AbstractTensor
from syft.workers import BaseWorker

logger = logging.getLogger(__name__)
TIMEOUT_INTERVAL = 9_999_999


class WebsocketClientWorker(BaseWorker):
    def __init__(
        self,
        hook,
        host: str,
        port: int,
        secure: bool = False,
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

        # creates the connection with the server which gets held open until the
        # WebsocketClientWorker is garbage collected.

        # Secure flag adds a secure layer applying cryptography and authentication
        self.uri = f"ws://{self.host}:{self.port}"
        if secure:
            self.uri = f"wss://{self.host}:{self.port}"
            ssl_settings = {"cert_reqs": ssl.CERT_NONE}
            self.ws = websocket.create_connection(
                self.uri, sslopt=ssl_settings, max_size=None, timeout=TIMEOUT_INTERVAL
            )
        else:
            # Insecure flow
            # Also avoid the server from timing out on the server-side in case of slow clients
            self.ws = websocket.create_connection(self.uri, max_size=None, timeout=TIMEOUT_INTERVAL)

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
            time.sleep(0.1)
            # Avoid timing out on the server-side
            self.ws = websocket.create_connection(self.uri, max_size=None, timeout=TIMEOUT_INTERVAL)
            logger.warning("Created new websocket connection")
            time.sleep(0.1)
            response = self._receive_action(message)
            if not self.ws.connected:
                raise RuntimeError(
                    "Websocket connection closed and creation of new connection failed."
                )
        return response

    def _send_msg_and_deserialize(self, command_name: str, *args, **kwargs):
        message = self.create_message_execute_command(
            command_name=command_name, command_owner="self", *args, **kwargs
        )

        # Send the message and return the deserialized response.
        serialized_message = sy.serde.serialize(message)
        response = self._recv_msg(serialized_message)
        return sy.serde.deserialize(response)

    def list_objects_remote(self):
        return self._send_msg_and_deserialize("list_objects")

    def fit(self, dataset_key):
        return self._send_msg_and_deserialize("fit", dataset=dataset_key)

    def fit_batch_remote(self):
        return self._send_msg_and_deserialize("fit_batch")

    def objects_count_remote(self):
        return self._send_msg_and_deserialize("objects_count")

    def __str__(self):
        """Returns the string representation of a Websocket worker.

        A to-string method for websocket workers that includes information from the websocket server

        Returns:
            The Type and ID of the worker

        """
        out = "<"
        out += str(type(self)).split("'")[1].split(".")[-1]
        out += " id:" + str(self.id)
        out += " #objects local:" + str(len(self._objects))
        out += " #objects remote: " + self.list_objects_remote()
        out += ">"
        return out
