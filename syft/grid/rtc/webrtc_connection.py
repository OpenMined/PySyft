import asyncio
import json

import syft as sy
from syft.codes import MSG_FIELD, GRID_EVENTS, NODE_EVENTS
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import (
    CopyAndPasteSignaling,
    object_to_string,
    object_from_string,
)
import threading
import queue
import time
from syft.workers.base import BaseWorker
from syft.messaging.message import SearchMessage
from syft.exceptions import GetNotPermittedError


class WebRTCConnection(threading.Thread, BaseWorker):

    OFFER = 1
    ANSWER = 2
    HOST_REQUEST = b"01"
    REMOTE_REQUEST = b"02"

    def __init__(self, grid_descriptor, worker, destination, connections, conn_type):
        """Create a new webrtc peer connection.

        Args:
            grid_descriptor: Grid network's websocket descriptor to forward webrtc
                            connection request.
            worker: Virtual Worker that represents this peer.
            destination: Destination Peer ID.
            connections: Peer connection descriptors.
            conn_type: Connection responsabilities this peer should provide. (offer, answer)
        """
        threading.Thread.__init__(self)
        BaseWorker.__init__(self, hook=sy.hook, id=destination)
        self._conn_type = conn_type
        self._origin = worker.id
        self._worker = worker
        self._worker.tensor_requests = []
        self._destination = destination
        self._grid = grid_descriptor
        self._msg = ""
        self._request_pool = queue.Queue()
        self._response_pool = queue.Queue()
        self.channel = None
        self.available = True
        self.connections = connections

    def _send_msg(self, message: bin, location=None):
        """Add a new syft operation on the request_pool to be processed asynchronously.

        Args:
            message : Binary Syft message.
            location : peer location (This parameter should be preserved to keep the
            BaseWorker compatibility, but we do not use it.)

        Returns:
            response_message: Binary Syft response message.
        """
        self._request_pool.put(WebRTCConnection.HOST_REQUEST + message)

        # Wait
        # PySyft is a sync library and should wait for this response.
        while self._response_pool.empty():
            time.sleep(0)
        return self._response_pool.get()

    def _recv_msg(self, message: bin):
        """Called when someone call syft function locally eg. tensor.send(node)

        PS: This method should be synchronized to keep the compatibility with Syft
        internal operations.
        Args:
            message: Binary Syft message.

        Returns:
            response_message : Binary syft response message.
        """
        if self.available:
            return self._send_msg(message)
        else:  # PySyft's GC delete commands
            return self._worker._recv_msg(message)

    # Running async all time
    async def send(self, channel):
        """Async method that will listen peer remote's requests and put it into the
        request_pool queue to be processed.

        Args:
            channel: Connection channel used by the peers.
        """
        while self.available:
            if not self._request_pool.empty():
                channel.send(self._request_pool.get())
            await asyncio.sleep(0)

    # Running async all time
    def process_msg(self, message, channel):
        """Process syft messages forwarding them to the peer virtual worker and put the
        response into the response_pool queue to be delivered async.

        Args:
            message: Binary syft message.
            channel: Connection channel used by the peers.
        """
        if message[:2] == WebRTCConnection.HOST_REQUEST:
            try:
                decoded_response = self._worker._recv_msg(message[2:])
            except GetNotPermittedError as e:
                message = sy.serde.deserialize(message[2:], worker=self._worker)
                self._worker.tensor_requests.append(message)
                decoded_response = sy.serde.serialize(e)

            channel.send(WebRTCConnection.REMOTE_REQUEST + decoded_response)
        else:
            self._response_pool.put(message[2:])

    def search(self, query):
        """Node's dataset search method overwrite.

        Args:
            query: Query used to search by the desired dataset tag.
        Returns:
            query_response: Return the peer's response.
        """
        message = SearchMessage(query)
        serialized_message = sy.serde.serialize(message)
        response = self._send_msg(serialized_message)
        return sy.serde.deserialize(response)

    # Main
    def run(self):
        """ Main thread method used to set up the connection and manage all the process."""
        self.signaling = CopyAndPasteSignaling()
        self.pc = RTCPeerConnection()

        if self._conn_type == WebRTCConnection.OFFER:
            func = self._set_offer
        else:
            func = self._run_answer

        self.loop = asyncio.new_event_loop()
        try:
            self.loop.run_until_complete(func(self.pc, self.signaling))
        except Exception:
            self.loop.run_until_complete(self.pc.close())

            # Stop loop:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    # OFFER
    async def _set_offer(self, pc, signaling):
        """Private method used to set up an offer to estabilish a new webrtc connection.

        Args:
            pc: Peer Connection  descriptor
            signaling: Webrtc signaling instance.
        """
        await signaling.connect()
        channel = pc.createDataChannel("chat")

        self.channel = channel

        @channel.on("open")
        def on_open():
            asyncio.ensure_future(self.send(channel))

        @channel.on("message")
        def on_message(message):
            self.process_msg(message, channel)

        await pc.setLocalDescription(await pc.createOffer())
        local_description = object_to_string(pc.localDescription)

        response = {
            MSG_FIELD.TYPE: NODE_EVENTS.WEBRTC_OFFER,
            MSG_FIELD.PAYLOAD: local_description,
            MSG_FIELD.FROM: self._origin,
        }

        forward_payload = {
            MSG_FIELD.TYPE: GRID_EVENTS.FORWARD,
            MSG_FIELD.DESTINATION: self._destination,
            MSG_FIELD.CONTENT: response,
        }

        self._grid.send(json.dumps(forward_payload))
        await self.consume_signaling(pc, signaling)

    # ANSWER
    async def _run_answer(self, pc, signaling):
        """Private method used to set up an answer to estabilish a new webrtc connection.

        Args:
            pc: Peer connection.
            signaling: Webrtc signaling instance.
        """
        await signaling.connect()

        @pc.on("datachannel")
        def on_datachannel(channel):
            asyncio.ensure_future(self.send(channel))

            self.channel = channel

            @channel.on("message")
            def on_message(message):
                self.process_msg(message, channel)

        await self.consume_signaling(pc, signaling)

    async def consume_signaling(self, pc, signaling):
        """Consume signaling to go through all the webrtc connection protocol.

        Args:
            pc: Peer Connection.
            signaling: Webrtc signaling instance.
        Exception:
            ConnectionClosedException: Exception used to finish this connection
            and close this thread.
        """
        # Async keep-alive connection thread
        while self.available:
            sleep_time = 0
            if self._msg == "":
                await asyncio.sleep(sleep_time)
                continue

            obj = object_from_string(self._msg)

            if isinstance(obj, RTCSessionDescription):
                await pc.setRemoteDescription(obj)
                if obj.type == "offer":
                    # send answer
                    await pc.setLocalDescription(await pc.createAnswer())
                    local_description = object_to_string(pc.localDescription)

                    response = {
                        MSG_FIELD.TYPE: NODE_EVENTS.WEBRTC_ANSWER,
                        MSG_FIELD.FROM: self._origin,
                        MSG_FIELD.PAYLOAD: local_description,
                    }

                    forward_payload = {
                        MSG_FIELD.TYPE: GRID_EVENTS.FORWARD,
                        MSG_FIELD.DESTINATION: self._destination,
                        MSG_FIELD.CONTENT: response,
                    }
                    self._grid.send(json.dumps(forward_payload))
                    sleep_time = 10
            self._msg = ""
        raise Exception

    def disconnect(self):
        """ Disconnect from the peer and finish this thread. """
        self.available = False
        del self.connections[self._destination]

    def set_msg(self, content: str):
        self._msg = content
