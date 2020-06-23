import asyncio
import json
import syft as sy
from syft.codes import MSG_FIELD, GRID_EVENTS, NODE_EVENTS
from syft.grid.peer import WebRTCPeer
from syft.exceptions import GetNotPermittedError
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import (
    object_to_string,
    object_from_string,
)
import queue
import weakref
import uvloop


class WebRTCManager:

    HOST_REQUEST = b"01"
    REMOTE_REQUEST = b"02"

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

        self._request_pool = asyncio.Queue(loop=self.loop)
        self._response_pool = asyncio.Queue(loop=self.loop)
        self._buffered_msg = {}

    @property
    def nodes(self):
        return list(self._connections.keys())

    def get(self, node_id: str):
        return self._connections.get(node_id, None)

    async def process_msg(self, message, channel, conn_obj):
        """ Process syft messages forwarding them to the peer virtual worker and put the
            response into the response_pool queue to be delivered async.

            Args:
                message: Binary syft message.
                channel: Connection channel used by the peers.
        """
        if message[:2] == WebRTCManager.HOST_REQUEST:
            try:
                decoded_response = self.worker._recv_msg(message[2:])
            except GetNotPermittedError as e:
                message = sy.serde.deserialize(message[2:], worker=self.worker)
                self.worker.tensor_requests.append(message)
                decoded_response = sy.serde.serialize(e)

            channel.send(WebRTCManager.REMOTE_REQUEST + decoded_response)
        elif message[:2] == WebRTCPeer.REMOTE_REQUEST:
            await self._response_pool.put(message[2:])
        elif message[:2] == WebRTCPeer.BUFFERED_REQUEST:
            if self._buffered_msg.get(conn_obj.id, None):
                self._buffered_msg[conn_obj.id] += message[2:]
            else:
                self._buffered_msg[conn_obj.id] = message[2:]
        else:
            self._buffered_msg[conn_obj.id] += message[2:]
            decoded_response = self.worker._recv_msg(self._buffered_msg[conn_obj.id])
            channel.send(WebRTCManager.REMOTE_REQUEST + decoded_response)

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
