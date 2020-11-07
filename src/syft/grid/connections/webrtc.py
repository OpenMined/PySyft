"""
WebRTC connection representation

 This class aims to represent a generic and
asynchronous peer-to-peer WebRTC connection
based in Syft BidirectionalConnection Inferface.

 This connection interface provides a full-duplex
channel, allowing this class to work as a client
and as a server at the same time.

 This class is useful to send/compute data
using a p2p channel without the need
of deploying servers on cloud providers
or settting firewalls rules  in order
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
    any offer msg in his queue.

    4 - The Signaling Server will check the existence of offer messages addressed
    to the PySyft Peer (Answer) made by the desired node address (PySyft.Address).
    If that's the case,so the offer message will be sent to the peer as a response.

    5 - [PUSH] The PySyft Peer (Answer) will process the offer message
    in order to know the network address of the other peer.
    If no problems were found, the answer peer will generate its own local
    description (IP, Mac address,  etc) and send it to the signaling server,
    addressing (PySyft.Address) it to the offer node as an AnswerMessage.

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
import random
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
from loguru import logger

# syft relative
from ...core.common.message import SignedEventualSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.serde.deserialize import _deserialize
from ...core.io.address import Address
from ...core.io.connection import BidirectionalConnection
from ...core.node.abstract.node import AbstractNode
from ...decorators.syft_decorator_impl import syft_decorator
from ..services.signaling_service import CloseConnectionMessage

try:
    # stdlib
    from asyncio import get_running_loop  # noqa Python >=3.7
except ImportError:  # pragma: no cover
    # stdlib
    from asyncio.events import _get_running_loop as get_running_loop  # pragma: no cover


message_cooldown = 0.0


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
        try:
            self.loop = get_running_loop()
            log = "♫♫♫ > ...using a running event loop..."
            logger.debug(log)
            print(log)
        except RuntimeError as e:
            self.loop = None
            log = f"♫♫♫ > ...error getting a running event Loop... {e}"
            logger.error(log)
            print(log)

        if self.loop is None:
            log = "♫♫♫ > ...creating a new event loop..."
            print(log)
            logger.debug(log)
            self.loop = asyncio.new_event_loop()

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
            self.channel: Optional[RTCDataChannel] = None
            self._client_address: Optional[Address] = None

            # asyncio.ensure_future(self.heartbeat())

        except Exception as e:
            log = f"Got an exception in WebRTCConnection __init__. {e}"
            logger.error(log)
            raise e

    @syft_decorator(typechecking=True)
    async def _set_offer(self) -> str:
        """Initialize a Real Time Communication Data Channel,
        set datachannel callbacks/tasks, and send offer payload
        message.

        :return: returns a signaling offer payload containing local description.
        :rtype: str
        """
        try:
            # Use the Peer Connection structure to
            # set the channel as a RTCDataChannel.
            self.channel = self.peer_connection.createDataChannel("datachannel")

            # This method will be called by as a callback
            # function by the aioRTC lib when the when
            # the connection opens.
            @self.channel.on("open")
            async def on_open() -> None:  # type : ignore
                self.__producer_task = asyncio.ensure_future(self.producer())

            # This method is the aioRTC "consumer" task
            # and will be running as long as connection remains.
            # At this point we're just setting the method behavior
            # It'll start running after the connection opens.
            @self.channel.on("message")
            async def on_message(message: Union[bin, str]) -> None:  # type: ignore
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
            log = f"Got an exception in WebRTCConnection _set_offer. {e}"
            logger.error(log)
            raise e

    @syft_decorator(typechecking=True)
    async def _set_answer(self, payload: str) -> str:
        """Receives a signaling offer payload, initialize/set
        datachannel callbacks/tasks, updates remote local description
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

                @self.channel.on("message")
                async def on_message(message: Union[bin, str]) -> None:  # type: ignore
                    await self.consumer(msg=message)

            return await self._process_answer(payload=payload)
        except Exception as e:
            log = f"Got an exception in WebRTCConnection _set_answer. {e}"
            logger.error(log)
            raise e

    @syft_decorator(typechecking=True)
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
            log = f"Got an exception in WebRTCConnection _process_answer. {e}"
            logger.error(log)
            raise e
        return None

    @syft_decorator(typechecking=True)
    async def producer(self) -> None:
        # Async task to send messages to the other side.
        # These messages will be enqueued by PySyft Node Clients
        # by using PySyft routes and ClientConnection's inheritance.
        try:
            while True:
                # If self.producer_pool is empty
                # give up task queue priority, giving
                # computing time to the next task.
                msg = await self.producer_pool.get()

                await asyncio.sleep(message_cooldown)
                # If self.producer_pool.get() returned a message
                # send it as a binary using the RTCDataChannel.
                # logger.critical(f"> Sending MSG {msg.message} ID: {msg.id}")
                self.channel.send(msg.to_bytes())  # type: ignore
        except Exception as e:
            log = f"Got an exception in WebRTCConnection producer. {e}"
            logger.error(log)
            raise e

    def close(self) -> None:
        try:
            # Build Close Message to warn the other peer
            bye_msg = CloseConnectionMessage(address=Address())

            self.channel.send(bye_msg.to_bytes())  # type: ignore

            # Finish async tasks related with this connection
            self._finish_coroutines()
        except Exception as e:
            log = f"Got an exception in WebRTCConnection close. {e}"
            logger.error(log)
            raise e

    def _finish_coroutines(self) -> None:
        try:
            asyncio.run(self.peer_connection.close())
            self.__producer_task.cancel()
        except Exception as e:
            log = f"Got an exception in WebRTCConnection _finish_coroutines. {e}"
            logger.error(log)
            raise e

    @syft_decorator(typechecking=True)
    async def consumer(self, msg: bin) -> None:  # type: ignore
        try:
            # Async task to receive/process messages sent by the other side.
            # These messages will be sent by the other peer
            # as a service requests or responses for requests made by
            # this connection previously (ImmediateSyftMessageWithReply).

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
            log = f"Got an exception in WebRTCConnection consumer. {e}"
            logger.error(log)
            raise e

    @syft_decorator(typechecking=True)
    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        """Executes/Replies requests instantly.

        :return: returns an instance of SignedImmediateSyftMessageWithReply
        :rtype: SignedImmediateSyftMessageWithoutReply
        """
        # Execute node services now
        try:
            r = random.randint(0, 100000)
            logger.debug(
                f"> Before recv_immediate_msg_with_reply {r} {msg.message} {type(msg.message)}"
            )
            reply = self.node.recv_immediate_msg_with_reply(msg=msg)
            logger.debug(
                f"> After recv_immediate_msg_with_reply {r} {msg.message} {type(msg.message)}"
            )
            return reply
        except Exception as e:
            log = f"Got an exception in WebRTCConnection recv_immediate_msg_with_reply. {e}"
            logger.error(log)
            raise e

    @syft_decorator(typechecking=True)
    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        """ Executes requests instantly. """
        try:
            r = random.randint(0, 100000)
            logger.debug(
                f"> Before recv_immediate_msg_without_reply {r} {msg.message} {type(msg.message)}"
            )
            self.node.recv_immediate_msg_without_reply(msg=msg)
            logger.debug(
                f"> After recv_immediate_msg_without_reply {r} {msg.message} {type(msg.message)}"
            )
        except Exception as e:
            log = f"Got an exception in WebRTCConnection recv_immediate_msg_without_reply. {e}"
            logger.error(log)
            raise e

    @syft_decorator(typechecking=True)
    def recv_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        """ Executes requests eventually. """
        try:
            self.node.recv_eventual_msg_without_reply(msg=msg)
        except Exception as e:
            log = f"Got an exception in WebRTCConnection recv_eventual_msg_without_reply. {e}"
            logger.error(log)
            raise e

    @syft_decorator(typechecking=False)
    def send_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithReply:
        """Sends high priority messages and wait for their responses.

        :return: returns an instance of SignedImmediateSyftMessageWithReply.
        :rtype: SignedImmediateSyftMessageWithReply
        """
        try:
            return asyncio.run(self.send_sync_message(msg=msg))
        except Exception as e:
            log = f"Got an exception in WebRTCConnection send_immediate_msg_with_reply. {e}"
            logger.error(log)
            raise e

    @syft_decorator(typechecking=True)
    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        """" Sends high priority messages without waiting for their reply. """
        try:
            # asyncio.run(self.producer_pool.put_nowait(msg))
            self.producer_pool.put_nowait(msg)
            time.sleep(message_cooldown)
        except Exception as e:
            log = f"Got an exception in WebRTCConnection send_immediate_msg_without_reply. {e}"
            logger.error(log)
            raise e

    @syft_decorator(typechecking=True)
    def send_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        """" Sends low priority messages without waiting for their reply. """
        try:
            asyncio.run(self.producer_pool.put(msg))
            time.sleep(message_cooldown)
        except Exception as e:
            log = f"Got an exception in WebRTCConnection send_eventual_msg_without_reply. {e}"
            logger.error(log)
            raise e

    @syft_decorator(typechecking=True)
    async def send_sync_message(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        """Send sync messages generically.

        :return: returns an instance of SignedImmediateSyftMessageWithoutReply.
        :rtype: SignedImmediateSyftMessageWithoutReply
        """
        try:
            # To ensure the sequence of sending / receiving messages
            # it's necessary to keep only a unique reference for reading
            # inputs (producer) and outputs (consumer).
            r = random.randint(0, 100000)
            # To be able to perform this method synchronously (waiting for the reply)
            # without blocking async methods, we need to use queues.

            # Enqueue the message to be sent to the target.
            logger.debug(f"> Before send_sync_message producer_pool.put blocking {r}")
            # self.producer_pool.put_nowait(msg)
            await self.producer_pool.put(msg)
            logger.debug(f"> After send_sync_message producer_pool.put blocking {r}")

            # Wait for the response checking the consumer queue.
            logger.debug(
                f"> Before send_sync_message consumer_pool.get blocking {r} {msg}"
            )
            logger.debug(
                f"> Before send_sync_message consumer_pool.get blocking {r} {msg.message}"
            )
            # before = time.time()
            # timeout_secs = 15

            response = await self.consumer_pool.get()

            #  asyncio.run()
            # self.async_check(before=before, timeout_secs=timeout_secs, r=r)

            logger.debug(f"> After send_sync_message consumer_pool.get blocking {r}")
            return response
        except Exception as e:
            log = f"Got an exception in WebRTCConnection send_eventual_msg_without_reply. {e}"
            logger.error(log)
            raise e

    async def async_check(
        self, before: float, timeout_secs: int, r: float
    ) -> SignedImmediateSyftMessageWithoutReply:
        while True:
            await asyncio.sleep(message_cooldown)
            try:
                response = self.consumer_pool.get_nowait()
                return response
            except Exception as e:
                now = time.time()
                logger.debug(
                    f"> During send_sync_message consumer_pool.get blocking {r}. {e}"
                )
                if now - before > timeout_secs:
                    log = f"send_sync_message timeout {timeout_secs} {r}"
                    logger.critical(log)
                    raise Exception(log)

    # async def heartbeat(self) -> None:
    #     producer_watermark = 0
    #     consumer_watermark = 0
    #     while True:
    #         await asyncio.sleep(5)
    #         try:
    #             psize = self.producer_pool.qsize()
    #             csize = self.consumer_pool.qsize()
    #             async_task_count = len(asyncio.all_tasks())
    #             producer_watermark = max(producer_watermark, psize)
    #             consumer_watermark = max(consumer_watermark, csize)
    #             log = (
    #                 f"{self.node.name} PQ: {psize} / {producer_watermark} - "
    #                 + f"CQ: {csize} / {consumer_watermark} - AT: {async_task_count}"
    #             )
    #             print(log)
    #             logger.critical(log)

    #             if getattr(self.peer_connection, "_RTCPeerConnection__isClosed", False):
    #                 log = f"☠️ HEART BEAT DEAD! {self.node.name}"
    #                 print(log)
    #                 logger.critical(log)
    #             # else:
    #             #     log = f"❤️ HEART BEAT ALIVE! {self.node.name}"
    #             #     print(log)
    #             #     logger.critical(log)
    #         except Exception as e:
    #             log = f"HEART BEAT exception in {self.node.name}. {e}"
    #             print(log)
    #             logger.critical(log)
    #             raise Exception(log)
