import binascii
import time
from typing import List
from typing import Union

import socketio
import torch

import syft as sy
from syft.frameworks.torch.tensors.interpreters import AbstractTensor
from syft.workers import BaseWorker


class WebsocketGridClient(BaseWorker):
    """ Websocket Grid Client """

    def __init__(
        self,
        hook,
        addr: str,
        id: Union[int, str] = 0,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor, AbstractTensor]] = None,
    ):
        """
        Args:
            hook : a normal TorchHook object
            addr : the address this client connects to
            id : the unique id of the worker (string or int)
            log_msgs : whether or not all messages should be
                saved locally for later inspection.
            verbose : a verbose option - will print all messages
                sent/received to stdout
            data : any initial tensors the server should be
                initialized with (such as datasets)
        """
        self.uri = addr
        self.response_from_client = None
        self.wait_for_client_event = False

        # Creates the connection with the server
        self.__sio = socketio.Client()
        super().__init__(hook, id, data, is_client_worker, log_msgs, verbose)

        @self.__sio.on("/identity/")
        def check_identity(msg):
            if msg != "OpenGrid":
                raise PermissionError("App is not an OpenGrid app")

        @self.__sio.on("/cmd")
        def on_client_result(args):
            if log_msgs:
                print("Receiving result from client {}".format(args))
            # The server broadcasted the results from another client
            self.response_from_client = binascii.unhexlify(args[2:-1])
            # Tell the wait_for_client_event to clear up and continue execution
            self.wait_for_client_event = False

        @self.__sio.on("/connect-node")
        def connect_node(msg):
            if self.verbose:
                print("Connect Grid Node: ", msg)

    def _send_msg(self, message: bin) -> bin:
        raise NotImplementedError

    def _recv_msg(self, message: bin) -> bin:
        if self.__sio.eio.state != "connected":
            raise ConnectionError("Worker is not connected to the server")

        message = str(binascii.hexlify(message))
        # Sends the message to the server
        self.__sio.emit("/cmd", {"message": message})

        self.wait_for_client_event = True
        # Wait until the server gets back with a result or an ACK
        while self.wait_for_client_event:
            continue

        # Return the result
        if self.response_from_client == "ACK":
            # Empty result for the serialiser to continue
            return sy.serde.serialize(b"")
        return self.response_from_client

    def connect_grid_node(self, addr=str, id=str):
        self.__sio.emit("/connect-node", {"uri": addr, "id": id})

    def connect(self):
        self.__sio.connect(self.uri)
        self.__sio.emit("/set-grid-id", {"id": self.id})

    def disconnect(self):
        self.__sio.disconnect()
