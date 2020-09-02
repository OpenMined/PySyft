from syft.core.common.message import (
    SyftMessage,
    SignedEventualSyftMessageWithoutReply,
    SignedImmediateSyftMessageWithoutReply,
    SignedImmediateSyftMessageWithReply,
)
from syft.core.io.connection import BidirectionalConnection
from syft.decorators.syft_decorator_impl import syft_decorator
from syft.core.common.serde.deserialize import _deserialize
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import (
    object_to_string,
    object_from_string,
)

from syft.core.node.abstract.node import AbstractNode
from typing import Union

import asyncio


class WebRTCConnection(BidirectionalConnection):
    def __init__(self, node):
        
        self.node = node

        # EventLoop that manages async tasks (producer/consumer)
        # This structure is global and needs to be
        # defined beforehand.
        try:
            self.loop = asyncio.get_running_loop()
            print("> Using WebSocket Event Loop")
        except RuntimeError as e:
            self.loop = asyncio.new_event_loop()
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

        self.peer_connection = RTCPeerConnection()
        self.channel = None

    async def _set_offer(self) -> str:
        self.channel = self.peer_connection.createDataChannel("chat")
        
        @self.channel.on("open")
        async def on_open():
            asyncio.ensure_future(self.producer())

        @self.channel.on("message")
        async def on_message(message):
            await self.consumer(message)

        await self.peer_connection.setLocalDescription(
            await self.peer_connection.createOffer()
        )

        local_description = object_to_string(self.peer_connection.localDescription)

        return local_description

    async def _set_answer(self, payload: str) -> str:
        self.peer_connection

        @self.peer_connection.on("datachannel")
        def on_datachannel(channel):
            self.channel = channel
            
            asyncio.ensure_future(self.producer())

            @self.channel.on("message")
            async def on_message(message):
                await self.consumer(message)

        return await self._process_answer(payload)


    async def _process_answer(self, payload: str) -> Union[str,None]:
        msg = object_from_string(payload)

        if isinstance(msg, RTCSessionDescription):
            await self.peer_connection.setRemoteDescription(msg)

            if msg.type == "offer":
                await self.peer_connection.setLocalDescription(
                    await self.peer_connection.createAnswer()
                )
                local_description = object_to_string(
                    self.peer_connection.localDescription
                )
                return local_description

    async def producer(self):
        while True:
            msg = await self.producer_pool.get()
            self.channel.send(msg.json())

    async def consumer(self, msg: SyftMessage ) -> None:
        _msg = _deserialize(blob=msg, from_json=True)

        if _msg.address != self._client_address:
            # Immediate message with reply
            if isinstance(_msg, SignedImmediateSyftMessageWithReply):
                reply = self.recv_immediate_msg_with_reply(msg=_msg)
                await self.producer_pool.put(reply)

            elif isinstance(_msg, SignedImmediateSyftMessageWithoutReply):
                self.recv_immediate_msg_without_reply(msg=_msg)

            # Eventual message without reply
            else:
                self.recv_eventual_msg_without_reply(msg=_msg)
        
        else:
            await self.consumer_pool.put(_msg)


    @syft_decorator(typechecking=True)
    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        """Executes/Replies requests instantly.

        :return: returns an instance of ImmediateSyftMessageWithReply
        :rtype: ImmediateSyftMessageWithReply
        """
        # Execute node services now
        reply = self.node.recv_immediate_msg_with_reply(msg=msg)
        return reply

    @syft_decorator(typechecking=True)
    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        """ Executes requests instantly. """
        self.node.recv_immediate_msg_without_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def recv_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        """ Executes requests eventually. """
        self.node.recv_eventual_msg_without_reply(msg=msg)

    @syft_decorator(typechecking=False)
    def send_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithReply:
        """Sends high priority messages and wait for their responses.
        :return: returns an instance of ImmediateSyftMessageWithReply.
        """
        return asyncio.run(self.send_sync_message(msg=msg))

    @syft_decorator(typechecking=True)
    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        """" Sends high priority messages without waiting for their reply. """
        self.producer_pool.put_nowait(msg)

    @syft_decorator(typechecking=True)
    def send_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        """" Sends low priority messages without waiting for their reply. """
        self.producer_pool.put(msg)

    @syft_decorator(typechecking=True)
    async def send_sync_message(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        """ Send sync messages generically. """

        # To ensure the sequence of sending / receiving messages
        # it's necessary to keep only a unique reference for reading
        # inputs (producer) and outputs (consumer).

        # To be able to perform this method synchronously (waiting for the reply)
        # without blocking async methods, we need to use queues.

        # Enqueue the message to be sent to the target.
        self.producer_pool.put_nowait(msg)

        # Wait for the response checking the consumer queue.
        response = await self.consumer_pool.get()

        return response
