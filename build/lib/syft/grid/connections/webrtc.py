"""
WebRTC connection representation

This class aims to represent a generic and
asynchronous peer-to-peer WebRTC connection
based in Syft BidirectionalConnection Interface.

This connection interface provides a full-duplex
channel, allowing this class to work as a client
and as a server at the same time.

This class is useful to send/compute data
using a p2p channel without the need
of deploying servers on cloud providers
or setting firewalls rules  in order
turn this process "visible" to the world.

How does it work?
The WebRTC (Web Real-Time Communication)
is a protocol that uses a full-duplex channel
(same API as WebSockets), in order to provide
a high-quality RTC, and NAT traversal
networking technique in order to be able
to reach private addresses.

The WebRTC API includes no provisions
for signaling, the applications are in charge
of managing how the connections will be established.
In our use case, we'll be using ICE (Interactive
Connectivity Establishment) to establish these connections.

Signaling Components:
    - PyGrid Network App (Signaling Server)
    - PySyft Peer Process (Offer)
    - PySyft Peer Process (Answer)

Signaling Steps:

    1 - [PUSH] PySyft Peer (Offer) will send an offer msg
    to the Signaling Server containing its own local
    description (IP, MAC address, etc), addressing (PySyft.Address)
    it to the target node.

    2 - The Signaling Server (PyGrid Network) will
    receive the offer msg and store it in the target's
    offer request's queue.

    3 - [PULL] The PySyft Peer (Answer) will send a message to
    the Signaling Server checking if the desired node pushed
    any offer msg in their queue.

    4 - The Signaling Server will check the existence of offer messages addressed
    to the PySyft Peer (Answer) made by the desired node address (PySyft.Address).
    If that's the case, then the offer message will be sent to the peer as a response.

    5 - [PUSH] The PySyft Peer (Answer) will process the offer message
    in order to know the network address of the other peer.
    If no problems were found, the answering peer will generate its own local
    description (IP, Mac address,  etc) and send it to the signaling server,
    addressing (PySyft.Address) it to the offering node as an AnswerMessage.

    6 - The Signaling Server (PyGrid Network) will
    receive the answer msg and store it in the target's answer request's
    queue.

    7 - [PULL] The PySyft Peer (Offer) will send a message to
    the Signaling Server checking if the desired node pushed
    any answer msg in his queue.

    8 - The Signaling Server will check the existence of answer messages addressed
    to the PySyft Peer (Offer) made by the desired node address (PySyft.Address).
    If that's the case, so the answer message will be sent to the peer as a response.

    9 - The PySyft Peer (Offer) will process the answer message
    in order to know the network address of the other peer.
    If no problems were found, the connection channel will be established.
"""

# stdlib
import asyncio
import math
import os
import secrets
import time
from typing import Any
from typing import Optional
from typing import Union

# third party
from aiortc import RTCDataChannel
from aiortc import RTCPeerConnection
from aiortc import RTCSessionDescription
from aiortc.contrib.signaling import object_from_string
from aiortc.contrib.signaling import object_to_string

# syft relative
from ... import serialize
from ...core.common.event_loop import loop
from ...core.common.message import SignedEventualSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.serde.deserialize import _deserialize
from ...core.io.address import Address
from ...core.io.connection import BidirectionalConnection
from ...core.node.abstract.node import AbstractNode
from ...logger import debug
from ...logger import traceback_and_raise
from ...util import validate_type
from ..services.signaling_service import CloseConnectionMessage

DC_CHUNKING_ENABLED = True
DC_CHUNK_START_SIGN = b"<<<CHUNK START>>>"

try:
    DC_MAX_CHUNK_SIZE = int(os.environ["DC_MAX_CHUNK_SIZE"])
except KeyError:
    DC_MAX_CHUNK_SIZE = 2 ** 18

try:
    DC_MAX_BUFSIZE = int(os.environ["DC_MAX_BUFSIZE"])
except KeyError:
    DC_MAX_BUFSIZE = 2 ** 22


class OrderedChunk:
    def __init__(self, idx: int, data: bytes):
        self.idx = idx
        self.data = data

    def save(self) -> bytes:
        return self.idx.to_bytes(4, "big") + self.data

    @classmethod
    def load(cls, data: bytes) -> "OrderedChunk":
        idx = int.from_bytes(data[:4], "big")
        data = data[4:]
        return cls(idx, data)


class WebRTCConnection(BidirectionalConnection):
    loop: Any

    def __init__(self, node: AbstractNode) -> None:
        # WebRTC Connection representation

        # As we have a full-duplex connection,
        # it's necessary to use a node instance
        # inside of this connection. In order to
        # be able to process requests sent by
        # the other peer.
        # All the requests messages will be forwarded
        # to this node.
        self.node = node

        # EventLoop that manages async tasks (producer/consumer)
        # This structure is global and needs to be
        # defined beforehand.

        self.loop = loop
        # Message pool (High Priority)
        # These queues will be used to manage
        # async  messages.
        try:
            self.producer_pool: asyncio.Queue = asyncio.Queue(
                loop=self.loop,
            )  # Request Messages / Request Responses
            self.consumer_pool: asyncio.Queue = asyncio.Queue(
                loop=self.loop,
            )  # Request Responses

            # Initialize a PeerConnection structure
            self.peer_connection = RTCPeerConnection()

            # Set channel descriptor as None
            # This attribute will be used for external classes
            # in order to verify if the connection channel
            # was established.
            self.channel: RTCDataChannel
            self._client_address: Optional[Address] = None

        except Exception as e:
            traceback_and_raise(e)

    async def _set_offer(self) -> str:
        """
        Initialize a Real-Time Communication Data Channel,
        set data channel callbacks/tasks, and send offer payload
        message.

        :return: returns a signaling offer payload containing local description.
        :rtype: str
        """
        try:
            # Use the Peer Connection structure to
            # set the channel as a RTCDataChannel.
            self.channel = self.peer_connection.createDataChannel(
                "datachannel",
            )
            # Keep send buffer busy with chunks
            self.channel.bufferedAmountLowThreshold = 4 * DC_MAX_CHUNK_SIZE

            # This method will be called by aioRTC lib as a callback
            # function when the connection opens.
            @self.channel.on("open")
            async def on_open() -> None:  # type : ignore
                self.__producer_task = asyncio.ensure_future(self.producer())

            chunked_msg = []
            chunks_pending = 0

            # This method is the aioRTC "consumer" task
            # and will be running as long as connection remains.
            # At this point we're just setting the method behavior
            # It'll start running after the connection opens.
            @self.channel.on("message")
            async def on_message(raw: bytes) -> None:
                nonlocal chunked_msg, chunks_pending

                chunk = OrderedChunk.load(raw)
                message = chunk.data

                if message == DC_CHUNK_START_SIGN:
                    chunks_pending = chunk.idx
                    chunked_msg = [b""] * chunks_pending
                elif chunks_pending:
                    if chunked_msg[chunk.idx] == b"":
                        chunks_pending -= 1
                    chunked_msg[chunk.idx] = message
                    if chunks_pending == 0:
                        await self.consumer(msg=b"".join(chunked_msg))
                else:
                    # Forward all received messages to our own consumer method.
                    await self.consumer(msg=message)

            # Set peer_connection to generate an offer message type.
            await self.peer_connection.setLocalDescription(
                await self.peer_connection.createOffer()
            )

            # Generates the local description structure
            # and serialize it to string afterwards.
            local_description = object_to_string(self.peer_connection.localDescription)

            # Return the Offer local_description payload.
            return local_description
        except Exception as e:
            traceback_and_raise(e)

    async def _set_answer(self, payload: str) -> str:
        """
        Receives a signaling offer payload, initialize/set
        Data channel callbacks/tasks, updates remote local description
        using offer's payload message and returns a
        signaling answer payload.

        :return: returns a signaling answer payload containing local description.
        :rtype: str
        """

        try:

            @self.peer_connection.on("datachannel")
            def on_datachannel(channel: RTCDataChannel) -> None:
                self.channel = channel

                self.__producer_task = asyncio.ensure_future(self.producer())

                chunked_msg = []
                chunks_pending = 0

                @self.channel.on("message")
                async def on_message(raw: bytes) -> None:
                    nonlocal chunked_msg, chunks_pending

                    chunk = OrderedChunk.load(raw)
                    message = chunk.data
                    if message == DC_CHUNK_START_SIGN:
                        chunks_pending = chunk.idx
                        chunked_msg = [b""] * chunks_pending
                    elif chunks_pending:
                        if chunked_msg[chunk.idx] == b"":
                            chunks_pending -= 1
                        chunked_msg[chunk.idx] = message
                        if chunks_pending == 0:
                            await self.consumer(msg=b"".join(chunked_msg))
                    else:
                        await self.consumer(msg=message)

            result = await self._process_answer(payload=payload)
            return validate_type(result, str)

        except Exception as e:
            traceback_and_raise(e)
            raise Exception("mypy workaound: should not get here")

    async def _process_answer(self, payload: str) -> Union[str, None]:
        # Converts payload received by
        # the other peer in aioRTC Object
        # instance.
        try:
            msg = object_from_string(payload)

            # Check if Object instance is a
            # description of RTC Session.
            if isinstance(msg, RTCSessionDescription):

                # Use the target's network address/metadata
                # to set the remote description of this peer.
                # This will basically say to this peer how to find/connect
                # with to other peer.
                await self.peer_connection.setRemoteDescription(msg)

                # If it's an offer message type,
                # generates your own local description
                # and send it back in order to tell
                # to the other peer how to find you.
                if msg.type == "offer":
                    # Set peer_connection to generate an offer message type.
                    await self.peer_connection.setLocalDescription(
                        await self.peer_connection.createAnswer()
                    )

                    # Generates the local description structure
                    # and serialize it to string afterwards.
                    local_description = object_to_string(
                        self.peer_connection.localDescription
                    )

                    # Returns the answer peer's local description
                    return local_description
        except Exception as e:
            traceback_and_raise(e)
        return None

    async def producer(self) -> None:
        """
        Async task to send messages to the other side.
        These messages will be enqueued by PySyft Node Clients
        by using PySyft routes and ClientConnection's inheritance.
        """
        try:
            while True:
                # If self.producer_pool is empty, give up task queue priority
                # and give computing time to the next task.
                msg = await self.producer_pool.get()

                # If self.producer_pool.get() returns a message
                # send it as a binary using the RTCDataChannel.
                data = serialize(msg, to_bytes=True)
                data_len = len(data)

                if DC_CHUNKING_ENABLED and data_len > DC_MAX_CHUNK_SIZE:
                    chunk_num = 0
                    done = False
                    sent: asyncio.Future = asyncio.Future(loop=self.loop)

                    def send_data_chunks() -> None:
                        nonlocal chunk_num, data_len, done, sent
                        # Send chunks until buffered amount is big or we're done
                        while (
                            self.channel.bufferedAmount <= DC_MAX_BUFSIZE and not done
                        ):
                            start_offset = chunk_num * DC_MAX_CHUNK_SIZE
                            end_offset = min(
                                (chunk_num + 1) * DC_MAX_CHUNK_SIZE, data_len
                            )
                            chunk = data[start_offset:end_offset]
                            self.channel.send(OrderedChunk(chunk_num, chunk).save())
                            chunk_num += 1
                            if chunk_num * DC_MAX_CHUNK_SIZE >= data_len:
                                done = True
                                sent.set_result(True)

                        if not done:
                            # Set listener for next round of sending when buffer is empty
                            self.channel.once("bufferedamountlow", send_data_chunks)

                    chunk_count = math.ceil(data_len / DC_MAX_CHUNK_SIZE)
                    self.channel.send(
                        OrderedChunk(chunk_count, DC_CHUNK_START_SIGN).save()
                    )
                    send_data_chunks()
                    # Wait until all chunks are dispatched
                    await sent
                else:
                    self.channel.send(OrderedChunk(0, data).save())
        except Exception as e:
            traceback_and_raise(e)

    def close(self) -> None:
        try:
            # Build Close Message to warn the other peer
            bye_msg = CloseConnectionMessage(address=Address())

            self.channel.send(OrderedChunk(0, serialize(bye_msg, to_bytes=True)).save())

            # Finish async tasks related with this connection
            self._finish_coroutines()
        except Exception as e:
            traceback_and_raise(e)

    def _finish_coroutines(self) -> None:
        try:
            asyncio.run(self.peer_connection.close())
            self.__producer_task.cancel()
        except Exception as e:
            traceback_and_raise(e)

    async def consumer(self, msg: bytes) -> None:
        """
        Async task to receive/process messages sent by the other side.
        These messages will be sent by the other peer as a service requests or responses
        for requests made by this connection previously (ImmediateSyftMessageWithReply).
        """
        try:
            # Deserialize the received message
            _msg = _deserialize(blob=msg, from_bytes=True)

            # Check if it's NOT  a response generated by a previous request
            # made by the client instance that uses this connection as a route.
            # PS: The "_client_address" attribute will be defined during
            # Node Client initialization.
            if _msg.address != self._client_address:
                # If it's a new service request, route it properly
                # using the node instance owned by this connection.

                # Immediate message with reply
                if isinstance(_msg, SignedImmediateSyftMessageWithReply):
                    reply = self.recv_immediate_msg_with_reply(msg=_msg)
                    await self.producer_pool.put(reply)

                # Immediate message without reply
                elif isinstance(_msg, SignedImmediateSyftMessageWithoutReply):
                    self.recv_immediate_msg_without_reply(msg=_msg)

                elif isinstance(_msg, CloseConnectionMessage):
                    # Just finish async tasks related with this connection
                    self._finish_coroutines()

                # Eventual message without reply
                else:
                    self.recv_eventual_msg_without_reply(msg=_msg)

            # If it's true, the message will have the client's address as destination.
            else:
                await self.consumer_pool.put(_msg)

        except Exception as e:
            traceback_and_raise(e)

    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        """
        Executes/Replies requests instantly.

        :return: returns an instance of SignedImmediateSyftMessageWithReply
        :rtype: SignedImmediateSyftMessageWithoutReply
        """
        # Execute node services now
        try:
            r = secrets.randbelow(100000)
            debug(
                f"> Before recv_immediate_msg_with_reply {r} {msg.message} {type(msg.message)}"
            )
            reply = self.node.recv_immediate_msg_with_reply(msg=msg)
            debug(
                f"> After recv_immediate_msg_with_reply {r} {msg.message} {type(msg.message)}"
            )
            return reply
        except Exception as e:
            traceback_and_raise(e)

    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        """
        Executes requests instantly.
        """
        try:
            r = secrets.randbelow(100000)
            debug(
                f"> Before recv_immediate_msg_without_reply {r} {msg.message} {type(msg.message)}"
            )
            self.node.recv_immediate_msg_without_reply(msg=msg)
            debug(
                f"> After recv_immediate_msg_without_reply {r} {msg.message} {type(msg.message)}"
            )
        except Exception as e:
            traceback_and_raise(e)

    def recv_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        """
        Executes requests eventually.
        """
        try:
            self.node.recv_eventual_msg_without_reply(msg=msg)
        except Exception as e:
            traceback_and_raise(e)
            raise Exception("mypy workaound: should not get here")

    # TODO: fix this mypy madness
    def send_immediate_msg_with_reply(  # type: ignore
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithReply:
        """
        Sends high priority messages and wait for their responses.

        :return: returns an instance of SignedImmediateSyftMessageWithReply.
        :rtype: SignedImmediateSyftMessageWithReply
        """
        try:
            # properly fix this!
            return validate_type(
                asyncio.run(self.send_sync_message(msg=msg)),
                object,
            )
        except Exception as e:
            traceback_and_raise(e)
            raise Exception("mypy workaound: should not get here")

    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        """
        Sends high priority messages without waiting for their reply.
        """
        try:
            # asyncio.run(self.producer_pool.put_nowait(msg))
            self.producer_pool.put_nowait(msg)
        except Exception as e:
            traceback_and_raise(e)

    def send_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        """
        Sends low priority messages without waiting for their reply.
        """
        try:
            asyncio.run(self.producer_pool.put(msg))
        except Exception as e:
            traceback_and_raise(e)

    async def send_sync_message(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        """
        Send sync messages generically.

        :return: returns an instance of SignedImmediateSyftMessageWithoutReply.
        :rtype: SignedImmediateSyftMessageWithoutReply
        """
        try:
            # To ensure the sequence of sending / receiving messages
            # it's necessary to keep only a unique reference for reading
            # inputs (producer) and outputs (consumer).
            r = secrets.randbelow(100000)
            # To be able to perform this method synchronously (waiting for the reply)
            # without blocking async methods, we need to use queues.

            # Enqueue the message to be sent to the target.
            debug(f"> Before send_sync_message producer_pool.put blocking {r}")
            # self.producer_pool.put_nowait(msg)
            await self.producer_pool.put(msg)
            debug(f"> After send_sync_message producer_pool.put blocking {r}")

            # Wait for the response checking the consumer queue.
            debug(f"> Before send_sync_message consumer_pool.get blocking {r} {msg}")
            debug(
                f"> Before send_sync_message consumer_pool.get blocking {r} {msg.message}"
            )
            response = await self.consumer_pool.get()

            debug(f"> After send_sync_message consumer_pool.get blocking {r}")
            return response
        except Exception as e:
            traceback_and_raise(e)

    async def async_check(
        self, before: float, timeout_secs: int, r: float
    ) -> SignedImmediateSyftMessageWithoutReply:
        while True:
            try:
                response = self.consumer_pool.get_nowait()
                return response
            except Exception as e:
                now = time.time()
                debug(f"> During send_sync_message consumer_pool.get blocking {r}. {e}")
                if now - before > timeout_secs:
                    traceback_and_raise(
                        Exception(f"send_sync_message timeout {timeout_secs} {r}")
                    )
