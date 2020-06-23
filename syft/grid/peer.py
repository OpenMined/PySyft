import queue
import syft as sy
import time
import asyncio
from syft.workers.base import BaseWorker
from syft.workers.virtual import VirtualWorker
from syft.messaging.message import SearchMessage
from syft.exceptions import GetNotPermittedError
from aiortc import RTCPeerConnection
import sys
import enum


class Header:
    HOST_REQUEST = b"01"
    REMOTE_REQUEST = b"02"
    BUFFERED_REQUEST = b"03"
    FINISH_HOST_BUFF_REQUEST = b"04"
    FINISH_REMOTE_BUFF_REQUEST = b"05"


class WebRTCPeer(BaseWorker):
    def __init__(
        self,
        node_id: str,
        worker: VirtualWorker,
        response_pool: asyncio.Queue,
        buffer_chunk: int = 262144,
    ):
        BaseWorker.__init__(self, sy.hook, id=node_id)
        self._response_pool = response_pool
        self._worker = worker
        self.pc = RTCPeerConnection()
        self.channel = None
        self._buffer_chunk = buffer_chunk

    async def _send_msg(self, message: bin, location=None):
        # Small messages should be sent in just one payload.
        if sys.getsizeof(message) <= self._buffer_chunk:
            self.channel.send(Header.HOST_REQUEST + message)
            return await self._response_pool.get()
        else:  # Split large messages into several chunks sending them in sequence.
            for i in range(0, len(message), self._buffer_chunk):
                # If it's the last chunk, use FINISH BUFFER header and wait for response.
                if i + self._buffer_chunk >= len(message):
                    self.channel.send(
                        Header.FINISH_HOST_BUFF_REQUEST + message[i : i + self._buffer_chunk]
                    )
                    return await self._response_pool.get()
                else:  # If it's not the last chunk, just send without wait for any response.
                    self.channel.send(Header.BUFFERED_REQUEST + message[i : i + self._buffer_chunk])

    def _recv_msg(self, message: bin):
        # if self.available:
        return asyncio.run(self._send_msg(message))
        # else:  # PySyft's GC delete commands
        #    return self._worker._recv_msg(message)

    def search(self, query):
        """ Node's dataset search method overwrite.

            Args:
                query: Query used to search by the desired dataset tag.
            Returns:
                query_response: Return the peer's response.
        """
        message = SearchMessage(query)
        serialized_message = sy.serde.serialize(message)
        response = asyncio.run(self._send_msg(serialized_message))
        return sy.serde.deserialize(response)
