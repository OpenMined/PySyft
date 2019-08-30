import binascii
from typing import List
from typing import Union

import torch

import syft as sy
from syft.generic.tensor import AbstractTensor
from syft.workers import BaseWorker
from syft.federated import FederatedClient
from syft.codes import MSGTYPE


class WebsocketGridClient(BaseWorker, FederatedClient):
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

        # Unfortunately, socketio will throw an exception on import if it's in a
        # thread. This occurs when Flask is in development mode
        import socketio

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

        @self.__sio.on("/cmd-response")
        def on_client_result(args):
            if log_msgs:
                print("Receiving result from client {}".format(args))
            try:
                # The server broadcasted the results from another client
                self.response_from_client = binascii.unhexlify(args[2:-1])
            except:
                raise Exception(args)

            # Tell the wait_for_client_event to clear up and continue execution
            self.wait_for_client_event = False

        @self.__sio.on("/connect-node-response")
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
            self.__sio.sleep()

        # Return the result
        if self.response_from_client == "ACK":
            # Empty result for the serialiser to continue
            return sy.serde.serialize(b"")
        return self.response_from_client

    def connect_grid_node(self, addr: str, id: str, sleep_time=0.5):
        self.__sio.emit("/connect-node", {"uri": addr, "id": id})
        self.__sio.sleep(sleep_time)

    def search(self, *query):
        # Prepare a message requesting the websocket server to search among its objects
        message = sy.Message(MSGTYPE.SEARCH, query)
        serialized_message = sy.serde.serialize(message)

        # Send the message and return the deserialized response.
        response = self._recv_msg(serialized_message)
        return sy.serde.deserialize(response)

    def connect(self):
        if self.__sio.eio.state != "connected":
            self.__sio.connect(self.uri)
            self.__sio.emit("/set-grid-id", {"id": self.id})

    def disconnect(self):
        self.__sio.disconnect()
