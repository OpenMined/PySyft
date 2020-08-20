from syft.core.io.connection import BidirectionalConnection

import websockets
from websockets.client import WebSocketClientProtocol

import asyncio

from syft.core.common.serde import Serializable
from syft.decorators.syft_decorator_impl import syft_decorator
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.client import Client
from syft.core.common.serde import _deserialize

from syft.core.common.message import (
    EventualSyftMessageWithoutReply,
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
)

from ...core.node.domain.service import RequestService


class WebsocketConnection(BidirectionalConnection):
    def __init__(self, url: str, node: AbstractNode) -> None:
        # Websocket Connection representation

        # This class aims to implement a generic and
        # asynchronous peer-to-peer websocket connection
        # based in Syft BidirectionalConnection Inferface.

        # Using this class we expect to be able to work
        # as a client, sending requests to the other peers
        # and as a server receiving and processing messages,
        # from the other entities.

        # This class is useful in PyGrid context,
        # when you need to perform requests querying
        # by other nodes or datasets, but also need to be
        # notified when someone wants to establish
        # a connection with you.

        # Uniform Resource Location
        # Domain Address that we want to be connected.
        self.url = url

        # Node obj instance (Domain, Worker, ...)
        # used to process requests made by other peers.
        self.node = node

        # EventLoop that manages async tasks (producer/consumer)
        # This structure is global and needs to be
        # defined beforehand.
        try:
            self.loop = asyncio.get_running_loop()
            print("> Using WebSocket Event Loop")
        except RuntimeError as e:
            self.loop = asyncio.get_event_loop()
            print(f"Error getting Event Loop. {e}")
            print("> Creating WebSocket Event Loop")

        # Message pool (High Priority)
        # These queues will be used to manage
        # messages.
        self.producer_pool: asyncio.Queue = asyncio.Queue(
            loop=self.loop
        )  # Request Messages / Request Responses
        self.consumer_pool: asyncio.Queue = asyncio.Queue(
            loop=self.loop
        )  # Request Responses

        asyncio.run(self.get_metadata())
        asyncio.ensure_future(self.connect())

    @syft_decorator(typechecking=False)
    async def handler(self, websocket: WebSocketClientProtocol) -> None:
        """ Websocket Handler."""
        # Start,monitor and finish producer/consumer tasks.
        # Adds producer and cosumer tasks into
        # the eventloop queue to be executed
        # concurrently.
        consumer_task = asyncio.ensure_future(self.consumer_handler(websocket))
        producer_task = asyncio.ensure_future(self.producer_handler(websocket))

        # Wait until one of them finishes (Websocket connection closed)
        done, pending = await asyncio.wait(
            [consumer_task, producer_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel the peding one.
        for task in pending:
            task.cancel()

    @syft_decorator(typechecking=False)
    async def producer_handler(self, websocket: WebSocketClientProtocol) -> None:
        """ Producer Handler Task"""

        # Async task to retreive ImmediateSyftMessages
        # from producer_pool queue and send them to target.
        while self.connected:
            # If producer pool is empty (any immediate message to send), wait here.
            message = await self.producer_pool.get()

            await websocket.send(message.json())

    @syft_decorator(typechecking=False)
    async def consumer_handler(self, websocket: WebSocketClientProtocol) -> None:
        """ Consumer Handler Task"""

        # Async task to receive and process messages (any type)
        # sent by the target.

        # PS: The consumer method will be responsible for routing
        # the messages properly according to its types.

        # If consumer pool is empty (any message to receive), wait here.
        async for message in websocket:
            await self.consumer(message)

    @syft_decorator(typechecking=True)
    def consumer(self, request: Serializable) -> None:
        """ Routes received requests properly."""

        # We expect to receive different request types:
        # 1 - RequestService: Made by the other peer spontaneously.
        # 2 - RequestResponse: Just a response from a request made
        # by this peer previously.

        # Besides that, the RequestService can store different Message types:
        # 1 - ImmediateSyftMessageWithReply
        # 2 - ImmediateSyftMessageWithoutReply
        # 3 - EventualSyftMessageWithoutReply

        # We need to route all of them properly.

        request = _deserialize(blob=request, from_json=True)

        # New Request Service
        if isinstance(request, RequestService):

            # Immediate message with reply
            if isinstance(request, ImmediateSyftMessageWithReply):
                reply = self.recv_immediate_msg_with_reply(msg=request)
                self.producer_pool.put(reply)

            # Immediate message without reply
            elif isinstance(request, ImmediateSyftMessageWithoutReply):
                self.recv_immediate_msg_without_reply(msg=request)

            # Eventual message without reply
            else:
                self.recv_eventual_msg_without_reply(msg=request)

        # Request Response
        else:
            self.consumer_pool.put(request)  # Just add in consumer pool queue.

    @syft_decorator(typechecking=True)
    async def connect(self) -> None:
        async with websockets.connect(self.url) as websocket:
            self.connected = True
            await self.handler(websocket)

    @syft_decorator(typechecking=True)
    def recv_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply
    ) -> ImmediateSyftMessageWithReply:
        """ Executes/Replies requests instantly.

            :return: returns an instance of ImmediateSyftMessageWithReply
            :rtype: ImmediateSyftMessageWithReply
        """
        # Execute node services now
        reply = self.node.recv_immediate_msg_with_reply(msg)
        return reply

    @syft_decorator(typechecking=True)
    def recv_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:
        """ Executes requests instantly. """
        self.node.recv_immediate_msg_without_reply(msg)

    @syft_decorator(typechecking=True)
    def recv_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply
    ) -> None:
        """ Executes requests eventually. """
        self.node.recv_eventual_msg_without_reply(msg)

    @syft_decorator(typechecking=False)
    def send_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply
    ) -> ImmediateSyftMessageWithReply:
        """ Sends high priority messages and wait for their responses.
            :return: returns an instance of ImmediateSyftMessageWithReply.
        """
        return asyncio.run(self.send_sync_message(msg=msg))

    @syft_decorator(typechecking=True)
    def send_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:
        """" Sends high priority messages without waiting for their reply. """
        self.producer_pool.put_nowait(msg)

    @syft_decorator(typechecking=True)
    def send_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply
    ) -> None:
        """" Sends low priority messages without waiting for their reply. """
        self.producer_pool.put(msg)

    @syft_decorator(typechecking=True)
    async def send_sync_message(
        self, msg: ImmediateSyftMessageWithReply
    ) -> ImmediateSyftMessageWithReply:
        """ Send sync messages generically. """

        # To ensure the sequence of sending / receiving messages
        # it's necessary to keep only a unique reference for reading
        # inputs (producer) and outputs (consumer).

        # To be able to perform this method synchronously (waiting for the reply)
        # without blocking async methods, we need to use queues.

        # Enqueue the message to be sent to the target.
        await self.producer_pool.put(msg)

        # Wait for the response checking the consumer queue.
        response = await self.consumer_pool.get()

        return response

    @syft_decorator(typechecking=True)
    async def get_metadata(self) -> None:
        async with websockets.connect(self.url + "/metadata") as websocket:
            await websocket.send("Hello!")
            response = await websocket.recv()
            address, name, client_id = Client.deserialize_client_metadata_from_node(
                response
            )
            self.metadata = (address, name, client_id)
