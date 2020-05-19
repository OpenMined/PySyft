from webrtc_connections import WebRTCConnection
from syft.workers.base import BaseWorker
import asyncio


class WebRTCManager(BaseWorker):
    """ Class used to manage multiple webrtc peer connections in different threads. """

    def __init__(self, grid_descriptor, syft_worker):
        self._connections = {}
        self._grid = grid_descriptor
        self.worker = syft_worker

    @property
    def nodes(self):
        """ Return all the peer nodes connected directly with this peer."""
        return list(self._connections.keys())

    def _send_msg(self, message: bin, location):
        """ Forward a local syft request to the proper destination. """
        return asyncio.run(self._connection[location.id].send(message))

    def get(self, node_id: str):
        """ Return a peer connection reference by its ID. """
        return self._connections.get(node_id, None)

    def process_answer(self, destination: str, content: str):
        """ Set the webrtc connection answer message. """
        self._connections[destination].set_msg(content)

    def process_offer(self, destination: str, content: str):
        """ Create a thread to process a webrtc offer connection. """
        self._connections[destination] = WebRTCConnection(
            self._grid, self.worker, destination, self._connections, WebRTCConnection.ANSWER,
        )
        self._connections[destination].set_msg(content)
        self._connections[destination].start()

    def start_offer(self, destination: str):
        """ Create a new thread to offer a webrtc connection. """
        self._connections[destination] = WebRTCConnection(
            self._grid, self.worker, destination, self._connections, WebRTCConnection.OFFER,
        )
        self._connections[destination].start()
