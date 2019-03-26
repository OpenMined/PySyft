from typing import List
from typing import Union

import socketio
import torch
import syft as sy

from syft.frameworks.torch.tensors.interpreters import AbstractTensor
from syft.workers import BaseWorker


class WebsocketIOClientWorker(BaseWorker):
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
        WebsocketIOServerWorker and receive all responses back from the server.
        """

        self.port = port
        self.host = host
        self.uri = f"http://{self.host}:{self.port}"

        # creates the connection with the server
        self.sio = socketio.Client()
        self.sio.connect(self.uri)

        @self.sio.on("connect")
        def on_connect():
            print("I'm connected!")

        @self.sio.on("message")
        def on_message(data):
            print("I received a message!")

        @self.sio.on("my message")
        def on_message(data):
            print("I received a custom message!")

        @self.sio.on("disconnect")
        def on_disconnect():
            print("I'm disconnected!")

        super().__init__(hook, id, data, is_client_worker, log_msgs, verbose)

    def _send_msg(self, message: bin) -> bin:
        raise RuntimeError(
            "_send_msg should never get called on a ",
            "WebsocketClientWorker. Did you accidentally "
            "make hook.local_worker a WebsocketClientWorker?",
        )

    def _recv_msg(self, message: bin) -> bin:
        """Forwards a message to the WebsocketIOServerWorker"""
        self.sio.emit('message', message)  # Block and wait for the response
        return sy.serde.serialize('')
        #
        # self.sio.send(str(binascii.hexlify(message)))
        # response = binascii.unhexlify(self.sio.recv()[2:-1])
        # return response
