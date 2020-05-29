import queue
import syft as sy
import time
import asyncio
from syft.workers.base import BaseWorker
from syft.workers.virtual import VirtualWorker
from syft.messaging.message import SearchMessage
from syft.exceptions import GetNotPermittedError
from aiortc import RTCPeerConnection


class WebRTCPeer(BaseWorker):

    HOST_REQUEST = b"01"
    REMOTE_REQUEST = b"02"

    def __init__(self, node_id: str, worker: VirtualWorker, response_pool: queue.Queue):
        BaseWorker.__init__(self, sy.hook, id=node_id)
        self._response_pool = response_pool
        self._worker = worker
        self.pc = RTCPeerConnection()
        self.channel = None

    async def _send_msg(self, message: bin, location=None):
        self.channel.send(WebRTCPeer.HOST_REQUEST + message)

        # Wait
        # PySyft is a sync library and should wait for this response.
        while self._response_pool.empty():
            await asyncio.sleep(0.01)

        return self._response_pool.get()

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
