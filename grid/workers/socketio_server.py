import asyncio
import time
from threading import Thread
from typing import List
from typing import Union

import torch
from flask import Flask

import syft as sy
from syft.generic.tensor import AbstractTensor
from syft.workers.virtual import VirtualWorker
from flask_socketio import SocketIO


class WebsocketIOServerWorker(VirtualWorker):
    """ Objects of this class can act as a remote worker or as a plain socket IO.

    By adding a payload to the object it will execute it forwarding the messages to the participants in the setup.

    If no payload is added, this object will be a plain socketIO sitting between two clients that implement the
    protocol.
    """

    def __init__(
        self,
        hook,
        host: str,
        port: int,
        payload=None,
        id: Union[int, str] = 0,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor, AbstractTensor]] = None,
    ):
        """
        Args:
            hook (sy.TorchHook): a normal TorchHook object
            host (str): the host on which the server should be run
            port (int): the port on which the server should be run
            payload (function): a function containing a list of operations
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
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, allow_upgrades=False)
        self.response_from_client = None
        self._init_job_thread()
        self.clients = []
        self.wait_for_client_event = True
        self._payload = payload

        super().__init__(
            hook=hook, id=id, data=data, log_msgs=log_msgs, verbose=verbose
        )

        @self.socketio.on("connect")
        def on_client_connect():
            if self.verbose:
                print("New client established connection")

        @self.socketio.on("client_id")
        def on_client_id(args):
            # Register client id and execute the payload for it
            if args not in self.clients:
                self.clients.append(args)
                # If this server has been created with a payload, execute it
                if self._payload is not None:
                    if self.verbose:
                        print(f"Client {args} connected. Start executing payload")
                    self._start_payload()

        @self.socketio.on("message")
        def send_command(args):
            self.socketio.emit("message", args)

        @self.socketio.on("client_ack")
        def on_client_ack(args):
            if self._payload is not None:
                # The client just sent ACK to indicate this server that the operation was done
                self.response_from_client = "ACK"
                # Tell the wait_for_client_event to clear up and continue execution
                self.wait_for_client_event = False
            # Broadcast the ack
            self.socketio.emit("client_ack", args)

        @self.socketio.on("client_send_result")
        def on_client_result(args):
            if self._payload is not None:
                # The client sent the results
                self.response_from_client = args
                # Tell the wait_for_client_event to clear up and continue execution
                self.wait_for_client_event = False
            # Broadcast the result
            self.socketio.emit("client_send_result", args)

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port, debug=self.log_msgs)

    def _send_msg(self, message: bin) -> bin:
        raise RuntimeError(
            "_send_msg should never get called on a ",
            "WebsocketIOServerWorker. Did you accidentally "
            "make hook.local_worker a WebsocketIOServerWorker?",
        )

    def _recv_msg(self, message: bin) -> bin:
        """Forwards a message to the WebsocketIOClientWorker"""
        self.socketio.emit("message", message)  # Block and wait for the response
        # This Event will wait until its `set()` is invoked.
        # This will be done when a message from the client is received
        # Ideally this should be done with semaphores or events
        self.wait_for_client_event = True
        while self.wait_for_client_event:
            time.sleep(0.1)

        if self.response_from_client == "ACK":
            return sy.serde.serialize(b"")
        return self.response_from_client

    def _init_job_thread(self):
        # Create the new loop and worker thread
        self.worker_loop = asyncio.new_event_loop()
        worker = Thread(target=self._start_job_loop, args=(self.worker_loop,))
        # Start the thread
        worker.start()

    @staticmethod
    def _start_job_loop(loop):
        """Switch to new event loop and run forever"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _start_payload(self):
        self.worker_loop.call_soon_threadsafe(self._payload, self)

    def terminate(self):
        self.worker_loop.call_soon_threadsafe(self.worker_loop.stop)
