import asyncio
import binascii
import logging
import ssl
from typing import Union
from typing import List

import tblib.pickling_support
import torch
import websockets

import syft as sy
from syft.generic.abstract.tensor import AbstractTensor
from syft.workers.virtual import VirtualWorker

from syft.exceptions import EmptyCryptoPrimitiveStoreError
from syft.exceptions import GetNotPermittedError
from syft.exceptions import ResponseSignatureError

tblib.pickling_support.install()


class WebsocketServerWorker(VirtualWorker):
    def __init__(
        self,
        hook,
        host: str,
        port: int,
        id: Union[int, str] = 0,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor, AbstractTensor]] = None,
        loop=None,
        cert_path: str = None,
        key_path: str = None,
    ):
        """This is a simple extension to normal workers wherein
        all messages are passed over websockets. Note that because
        BaseWorker assumes a request/response paradigm, this worker
        enforces this paradigm by default.

        Args:
            hook (sy.TorchHook): a normal TorchHook object
            id (str or id): the unique id of the worker (string or int)
            log_msgs (bool): whether or not all messages should be
                saved locally for later inspection.
            verbose (bool): a verbose option - will print all messages
                sent/received to stdout
            host (str): the host on which the server should be run
            port (int): the port on which the server should be run
            data (dict): any initial tensors the server should be
                initialized with (such as datasets)
            loop: the asyncio event loop if you want to pass one in
                yourself
            cert_path: path to used secure certificate, only needed for secure connections
            key_path: path to secure key, only needed for secure connections
        """

        self.port = port
        self.host = host
        self.cert_path = cert_path
        self.key_path = key_path

        if loop is None:
            loop = asyncio.new_event_loop()

        # this queue is populated when messages are received
        # from a client
        self.broadcast_queue = asyncio.Queue()

        # this is the asyncio event loop
        self.loop = loop

        # call BaseWorker constructor
        super().__init__(hook=hook, id=id, data=data, log_msgs=log_msgs, verbose=verbose)

    async def _consumer_handler(self, websocket: websockets.WebSocketCommonProtocol):
        """This handler listens for messages from WebsocketClientWorker
        objects.

        Args:
            websocket: the connection object to receive messages from and
                add them into the queue.

        """
        try:
            while True:
                msg = await websocket.recv()
                await self.broadcast_queue.put(msg)
        except websockets.exceptions.ConnectionClosed:
            self._consumer_handler(websocket)

    async def _producer_handler(self, websocket: websockets.WebSocketCommonProtocol):
        """This handler listens to the queue and processes messages as they
        arrive.

        Args:
            websocket: the connection object we use to send responses
                back to the client.

        """
        while True:

            # get a message from the queue
            message = await self.broadcast_queue.get()

            # convert that string message to the binary it represent
            message = binascii.unhexlify(message[2:-1])

            # process the message
            response = self._recv_msg(message)

            # convert the binary to a string representation
            # (this is needed for the websocket library)
            response = str(binascii.hexlify(response))

            # send the response
            await websocket.send(response)

    def _recv_msg(self, message: bin) -> bin:
        try:
            return self.recv_msg(message)
        except (EmptyCryptoPrimitiveStoreError, GetNotPermittedError, ResponseSignatureError) as e:
            return sy.serde.serialize(e)

    async def _handler(self, websocket: websockets.WebSocketCommonProtocol, *unused_args):
        """Setup the consumer and producer response handlers with asyncio.

        Args:
            websocket: the websocket connection to the client

        """

        asyncio.set_event_loop(self.loop)
        consumer_task = asyncio.ensure_future(self._consumer_handler(websocket))
        producer_task = asyncio.ensure_future(self._producer_handler(websocket))

        done, pending = await asyncio.wait(
            [consumer_task, producer_task], return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()

    def start(self):
        """Start the server"""
        # Secure behavior: adds a secure layer applying cryptography and authentication
        if not (self.cert_path is None) and not (self.key_path is None):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.cert_path, self.key_path)
            start_server = websockets.serve(
                self._handler,
                self.host,
                self.port,
                ssl=ssl_context,
                max_size=None,
                ping_timeout=None,
                close_timeout=None,
            )
        else:
            # Insecure
            start_server = websockets.serve(
                self._handler,
                self.host,
                self.port,
                max_size=None,
                ping_timeout=None,
                close_timeout=None,
            )

        asyncio.get_event_loop().run_until_complete(start_server)
        print("Serving. Press CTRL-C to stop.")
        try:
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            logging.info("Websocket server stopped.")
