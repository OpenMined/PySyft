import asyncio
import json
import syft as sy
from syft.codes import MSG_FIELD, GRID_EVENTS, NODE_EVENTS
from syft.grid.peer import WebRTCPeer, Header
from syft.exceptions import GetNotPermittedError
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import (
    object_to_string,
    object_from_string,
)
import queue
import weakref
import uvloop
import sys


class WebRTCManager:
    def __init__(self, grid_descriptor, syft_worker):
        self._connections = {}
        self._grid = grid_descriptor

        self.worker = syft_worker

        # Uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

        self.loop = asyncio.get_event_loop()

        # If an external async process already exists (eg. ipython kernel / tornado features)
        if self.loop.is_running():
            self.async_method = self.loop.create_task
        else:
            self.async_method = asyncio.run

        self._response_pool = asyncio.Queue(loop=self.loop)
        self._buffered_msg = {}

    @property
    def nodes(self):
        return list(self._connections.keys())

    def get(self, node_id: str):
        return self._connections.get(node_id, None)

    async def _process_host_request(self, message, channel, conn_obj):
        try:
            message = self.worker._recv_msg(message)
        except GetNotPermittedError as e:
            message = sy.serde.deserialize(message, worker=self.worker)
            self.worker.tensor_requests.append(message)
            decoded_response = sy.serde.serialize(e)

        # Small messages should be sent in just one payload.
        if sys.getsizeof(message) <= conn_obj._buffer_chunk:
            channel.send(Header.REMOTE_REQUEST + message)
        else:  # Split large messages into several chunks sending them in sequence.
            for i in range(0, len(message), conn_obj._buffer_chunk):
                # If it's the last chunk, use FINISH BUFFER header and wait for response.
                if i + conn_obj._buffer_chunk >= len(message):
                    channel.send(
                        Header.FINISH_REMOTE_BUFF_REQUEST + message[i : i + conn_obj._buffer_chunk]
                    )
                else:  # If it's not the last chunk, just send without wait for any response.
                    channel.send(Header.BUFFERED_REQUEST + message[i : i + conn_obj._buffer_chunk])

    async def _process_host_buffer_payload(self, message, channel, conn_obj):
        self._buffered_msg[conn_obj.id] += message
        decoded_response = self.worker._recv_msg(self._buffered_msg[conn_obj.id])
        channel.send(Header.REMOTE_REQUEST + decoded_response)
        del self._buffered_msg[conn_obj.id]

    async def _process_remote_request(self, message, channel, conn_obj):
        await self._response_pool.put(message)

    async def _process_remote_buffer_payload(self, message, channel, conn_obj):
        self._buffered_msg[conn_obj.id] += message
        await self._response_pool.put(self._buffered_msg[conn_obj.id])
        del self._buffered_msg[conn_obj.id]

    async def _aggregate_buffered_payload(self, message, channel, conn_obj):
        if self._buffered_msg.get(conn_obj.id, None):
            self._buffered_msg[conn_obj.id] += message
        else:
            self._buffered_msg[conn_obj.id] = message

    async def process_msg(self, message, channel, conn_obj):
        """ Process syft messages forwarding them to the peer virtual worker and put the
            response into the response_pool queue to be delivered async.

            Args:
                message: Binary syft message.
                channel: Connection channel used by the peers.
        """
        self.message_routes = {
            Header.HOST_REQUEST: self._process_host_request,
            Header.REMOTE_REQUEST: self._process_remote_request,
            Header.BUFFERED_REQUEST: self._aggregate_buffered_payload,
            Header.FINISH_HOST_BUFF_REQUEST: self._process_host_buffer_payload,
            Header.FINISH_REMOTE_BUFF_REQUEST: self._process_remote_buffer_payload,
        }

        # All methods add their responses into the aiortc response queue which is also async, so we do not need to care about queuing responses.
        asyncio.run_coroutine_threadsafe(
            self.message_routes[message[:2]](message[2:], channel, conn_obj), self.loop
        )

    def process_answer(self, destination: str, content: str):
        asyncio.run_coroutine_threadsafe(self._process_answer(destination, content), self.loop)

    def process_offer(self, destination: str, content: str):
        """ Create a thread to process a webrtc offer connection. """
        self._connections[destination] = WebRTCPeer(
            destination, weakref.proxy(self.worker), weakref.proxy(self._response_pool)
        )
        asyncio.run_coroutine_threadsafe(self._set_answer(destination, content), self.loop)

    def start_offer(self, destination: str):
        """ Create a new thread to offer a webrtc connection. """
        self._connections[destination] = WebRTCPeer(
            destination, weakref.proxy(self.worker), weakref.proxy(self._response_pool)
        )
        asyncio.run_coroutine_threadsafe(self._set_offer(destination), self.loop)

    async def _process_answer(self, destination: str, content: str):
        """ Set the webrtc connection answer message. """
        webrtc_obj = self._connections[destination]

        msg = object_from_string(content)
        if isinstance(msg, RTCSessionDescription):
            await webrtc_obj.pc.setRemoteDescription(msg)
            if msg.type == "offer":
                # send answer
                await webrtc_obj.pc.setLocalDescription(await webrtc_obj.pc.createAnswer())
                local_description = object_to_string(webrtc_obj.pc.localDescription)

                response = {
                    MSG_FIELD.TYPE: NODE_EVENTS.WEBRTC_ANSWER,
                    MSG_FIELD.FROM: self.worker.id,
                    MSG_FIELD.PAYLOAD: local_description,
                }

                forward_payload = {
                    MSG_FIELD.TYPE: GRID_EVENTS.FORWARD,
                    MSG_FIELD.DESTINATION: destination,
                    MSG_FIELD.CONTENT: response,
                }
                self._grid.send(json.dumps(forward_payload))

    # OFFER
    async def _set_offer(self, destination: str):
        conn_obj = self._connections[destination]

        conn_obj.channel = conn_obj.pc.createDataChannel("chat")
        channel = conn_obj.channel

        @channel.on("message")
        async def on_message(message):
            await self.process_msg(message, channel, conn_obj)

        await conn_obj.pc.setLocalDescription(await conn_obj.pc.createOffer())
        local_description = object_to_string(conn_obj.pc.localDescription)

        response = {
            MSG_FIELD.TYPE: NODE_EVENTS.WEBRTC_OFFER,
            MSG_FIELD.PAYLOAD: local_description,
            MSG_FIELD.FROM: self.worker.id,
        }

        forward_payload = {
            MSG_FIELD.TYPE: GRID_EVENTS.FORWARD,
            MSG_FIELD.DESTINATION: destination,
            MSG_FIELD.CONTENT: response,
        }
        self._grid.send(json.dumps(forward_payload))

    # ANSWER
    async def _set_answer(self, destination: str, content):
        conn_obj = self._connections[destination]

        pc = conn_obj.pc

        @pc.on("datachannel")
        def on_datachannel(channel):

            self._connections[destination].channel = channel

            @channel.on("message")
            async def on_message(message):
                await self.process_msg(message, channel, conn_obj)

        await self._process_answer(destination, content)

    def __getitem__(self, key):
        """
        Args:
            key: Node ID

        Returns:
            Return a peer connection reference by its ID.
        """

        return self.get(key)

    def __len__(self):
        return len(self._connections)
