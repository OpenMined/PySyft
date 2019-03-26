import asyncio
import time
from threading import Thread
from typing import List
from typing import Union

import torch
from flask import Flask
from flask_socketio import SocketIO

import syft as sy
from syft.frameworks.torch.tensors.interpreters import AbstractTensor
from syft.workers.virtual import VirtualWorker


# Summary of what is going on here
# SocketIO events are executed in the thread that launches this object
# A second thread is created in by calling _init_job_thread() (would it be better to create them for each client)
# When a client connects to the socket a the payload is passed to this second thread
# PySyft _recv_msg will perform a socketio.emit('message', message) for each operation and then it will wait until
# the client responds via on_new_client_message. In this method, the semaphore used by _recv_msg to wait is lifted
# allowing it to continue by working with the result sent by the client


class WebsocketIOServerWorker(VirtualWorker):
    def __init__(
        self,
        hook,
        host: str,
        port: int,
        payload,
        id: Union[int, str] = 0,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor, AbstractTensor]] = None,
    ):

        self.port = port
        self.host = host
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.response_from_client = None
        self._init_job_thread()
        self.clients = []
        self.wait_for_client_event = True
        self._payload = payload

        super().__init__(hook=hook, id=id, data=data, log_msgs=log_msgs, verbose=verbose)

        @self.socketio.on("connect")
        def on_client_connect():
            print("New client established connection")

        @self.socketio.on("client_id")
        def on_client_id(args):
            # Register client id and execute the payload for it
            if args not in self.clients:
                print(f'Client {args} connected. Start executing payload')
                self.clients.append(args)
                self._start_payload()

        @self.socketio.on("message")
        def send_command(args):
            print("Sending command to whoever is listening {}".format(args))
            self.socketio.emit("message", args)

        @self.socketio.on("client_ack")
        def on_client_ack():
            # The client just sent ACK to indicate this server that the operation was done
            self.response_from_client = b''
            # Tell the wait_for_client_event to clear up and continue execution
            self.wait_for_client_event = False

        @self.socketio.on("client_send_result")
        def on_client_result(args):
            print("Receiving result§ from client {}".format(args))
            # The client just sent ACK to indicate this server that the operation was done
            self.response_from_client = args
            # Tell the wait_for_client_event to clear up and continue execution
            self.wait_for_client_event = False

    def run(self):
        self.socketio.run(self.app, host=self.host, port=self.port, debug=self.log_msgs)

    def _send_msg(self, message: bin) -> bin:
        raise RuntimeError(
            "_send_msg should never get called on a ",
            "WebsocketClientWorker. Did you accidentally "
            "make hook.local_worker a WebsocketClientWorker?",
        )

    def _recv_msg(self, message: bin) -> bin:
        """Forwards a message to the WebsocketServerWorker"""
        print('_recv_msg in Web socket Server')
        self.socketio.emit('message', message)  # Block and wait for the response
        print('Message sent to client. Waiting for its response')
        # This Event will wait until its `set()` is invoked.
        # This will be done when a message from the client is received
        # Ideally this should be done with semaphores or events
        self.wait_for_client_event = True
        while self.wait_for_client_event:
            time.sleep(0.1)

        if self.response_from_client == b'':
            return sy.serde.serialize(b'')
        return self.response_from_client

    def _init_job_thread(self):
        # Create the new loop and worker thread
        self.worker_loop = asyncio.new_event_loop()
        worker = Thread(target=self._start_job_loop, args=(self.worker_loop,))
        # Start the thread
        worker.start()
        pass

    @staticmethod
    def _start_job_loop(loop):
        """Switch to new event loop and run forever"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _start_payload(self):
        self.worker_loop.call_soon_threadsafe(self._payload, self)

