import threading
import websocket
import json
from syft.codes import NODE_EVENTS, GRID_EVENTS, MSG_FIELD
from syft.frameworks.torch.tensors.interpreters.private import PrivateTensor
from syft.grid.rtc.nodes_manager import WebRTCManager
from syft.grid.rtc.peer_events import (
    _monitor,
    _create_webrtc_scope,
    _accept_offer,
    _process_webrtc_answer,
)

import syft as sy
import time


class Network(threading.Thread):
    """Grid Network class to operate in background processing grid requests
    and handling multiple peer connections with different nodes.
    """

    # Events called by the grid monitor to health checking and signaling webrtc connections.
    EVENTS = {
        NODE_EVENTS.MONITOR: _monitor,
        NODE_EVENTS.WEBRTC_SCOPE: _create_webrtc_scope,
        NODE_EVENTS.WEBRTC_OFFER: _accept_offer,
        NODE_EVENTS.WEBRTC_ANSWER: _process_webrtc_answer,
    }

    def __init__(self, node_id: str, **kwargs):
        """Create a new thread to send/receive messages from the grid service.

        Args:
            node_id: ID used to identify this peer.
        """
        threading.Thread.__init__(self)
        self._connect(**kwargs)
        self._worker = self._update_node_infos(node_id)
        self._worker.models = {}
        self._connection_handler = WebRTCManager(self._ws, self._worker)
        self.available = False

    def run(self):
        """Run the thread sending a request to join into the grid network and listening
        the grid network requests.
        """

        # Join
        self._join()
        try:
            # Listen
            self._listen()
        except OSError:  # Avoid IO socket errors
            pass

    def stop(self):
        """ Finish the thread and disconnect with the grid network. """
        self.available = False
        self._ws.shutdown()

    def _update_node_infos(self, node_id: str):
        """Create a new virtual worker to store/compute datasets owned by this peer.

        Args:
            node_id: ID used to identify this peer.
        """
        worker = sy.VirtualWorker(sy.hook, id=node_id)
        sy.local_worker._known_workers[node_id] = worker
        sy.local_worker.is_client_worker = False
        return worker

    def _listen(self):
        """Listen the sockets waiting for grid network health checks and webrtc
        connection requests.
        """
        while self.available:
            message = self._ws.recv()
            msg = json.loads(message)
            response = self._handle_messages(msg)
            if response:
                self._ws.send(json.dumps(response))

    def _handle_messages(self, message):
        """Route and process the messages received from the websocket connection.

        Args:
            message : message to be processed.
        """
        msg_type = message.get(MSG_FIELD.TYPE, None)
        if msg_type in Network.EVENTS:
            return Network.EVENTS[msg_type](message, self._connection_handler)

    def _connect(self, **kwargs):
        """ Create a websocket connection between this peer and the grid network. """
        self._ws = websocket.create_connection(**kwargs)

    @property
    def id(self):
        return self._worker.id

    def connect(self, destination_id: str):
        """Create a webrtc connection between this peer and the destination peer by using the grid network
        to forward the webrtc connection request protocol.

        Args:
            destination_id : Id used to identify the peer to be connected.
        """

        # Temporary Notebook async weird constraints
        # Should be removed after solving #3572
        if len(self._connection_handler) >= 1:
            print(
                "Due to some jupyter notebook async constraints, we do not recommend handling "
                "multiple connection peers at the same time."
            )
            print("This issue is in WIP status and may be solved soon.")
            print(
                "You can follow its progress here: https://github.com/OpenMined/PySyft/issues/3572"
            )
            return None

        webrtc_request = {MSG_FIELD.TYPE: NODE_EVENTS.WEBRTC_SCOPE, MSG_FIELD.FROM: self.id}

        forward_payload = {
            MSG_FIELD.TYPE: GRID_EVENTS.FORWARD,
            MSG_FIELD.DESTINATION: destination_id,
            MSG_FIELD.CONTENT: webrtc_request,
        }

        self._ws.send(json.dumps(forward_payload))
        while not self._connection_handler.get(destination_id):
            time.sleep(1)

        return self._connection_handler.get(destination_id)

    def disconnect(self, destination_id: str):
        """Disconnect with some peer connected previously.

        Args:
            destination_id: Id used to identify the peer to be disconnected.
        """
        _connection = self._connection_handler.get(destination_id)
        if _connection:
            _connection.available = False

    def host_dataset(self, dataset):
        """Host dataset using the virtual worker defined previously.

        Args:
            dataset: Dataset to be hosted.
        """
        allowed_users = None

        # By default the peer should be allowed to access its own private tensors.
        if dataset.is_wrapper and type(dataset.child) == PrivateTensor:
            dataset.child.register_credentials([self._worker.id])

        return dataset.send(self._worker, user=self._worker.id)

    def host_model(self, model):
        """ Host model using the virtual worker defined previously. """
        model.nodes.append(self._worker.id)
        self._worker.models[model.id] = model
        return model._model

    def _join(self):
        """ Send a join requet to register this peer on the grid network. """
        # Join into the network
        join_payload = {MSG_FIELD.TYPE: GRID_EVENTS.JOIN, MSG_FIELD.NODE_ID: self._worker.id}
        self._ws.send(json.dumps(join_payload))
        response = json.loads(self._ws.recv())
        self.available = True
        return response

    def __repr__(self):
        """Default String representation"""
        return (
            f"< Peer ID: {self.id}, "
            f"hosted datasets: {list(self._worker.object_store._tag_to_object_ids.keys())}, "
            f"hosted_models: {list(self._worker.models.keys())}, "
            f"connected_nodes: {list(self._connection_handler.nodes)}"
        )

    @property
    def peers(self):
        """
        Get WebRTCManager object
        """
        return self._connection_handler
