import time
from typing import List
from typing import Union

import torch

import syft as sy
from syft.frameworks.torch.tensors.interpreters import AbstractTensor
from syft.workers.virtual import VirtualWorker
import socketio

class WebsocketIOClientWorker(VirtualWorker):
    """A worker that forwards a message to a SocketIO server and wait for its response.

    This client then waits until the server returns with a result or an ACK at which point it finishes the
    _recv_msg operation.
    """

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
        """
        Args:
            hook (sy.TorchHook): a normal TorchHook object
            host (str): the host this client connects to
            port (int): the port this client connects to
            id (str or id): the unique id of the worker (string or int)
            log_msgs (bool): whether or not all messages should be
                saved locally for later inspection.
            verbose (bool): a verbose option - will print all messages
                sent/received to stdout
            data (dict): any initial tensors the server should be
                initialized with (such as datasets)
        """

        self.port = port
        self.host = host
        self.uri = f"http://{self.host}:{self.port}"

        self.response_from_client = None
        self.wait_for_client_event = False

        # Creates the connection with the server
        self.sio = socketio.Client()
        super().__init__(hook, id, data, is_client_worker, log_msgs, verbose)

        @self.sio.on("client_ack")
        def on_client_ack(args):
            # The server broadcasted an ACK from another client
            self.response_from_client = "ACK"
            # Tell the wait_for_client_event to clear up and continue execution
            self.wait_for_client_event = False

        @self.sio.on("client_send_result")
        def on_client_result(args):
            if log_msgs:
                print("Receiving result from client {}".format(args))
            # The server broadcasted the results from another client
            self.response_from_client = args
            # Tell the wait_for_client_event to clear up and continue execution
            self.wait_for_client_event = False

    def _send_msg(self, message: bin) -> bin:
        raise RuntimeError(
            "_send_msg should never get called on a ",
            "WebsocketIOClientWorker. Did you accidentally "
            "make hook.local_worker a WebsocketIOClientWorker?",
        )

    def _recv_msg(self, message: bin) -> bin:
        # Sends the message to the server
        self.sio.emit("message", message)
        self.wait_for_client_event = True
        # Wait until the server gets back with a result or an ACK
        while self.wait_for_client_event:
            time.sleep(0.1)

        # Return the result
        if self.response_from_client == "ACK":
            # Empty result for the serialiser to continue
            return sy.serde.serialize(b"")
        return self.response_from_client

    def connect(self):
        self.sio.connect(self.uri)
        self.sio.emit("client_id", self.id)

    def disconnect(self):
        self.sio.disconnect()
