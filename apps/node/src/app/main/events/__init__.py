"""This file exists to provide a route to websocket events."""
# Standard Python imports
import json

# External imports
from syft.codes import REQUEST_MSG

from .. import ws
from ..core.codes import *
from .data_centric.control_events import *
from .data_centric.model_events import *
from .data_centric.syft_events import *
from .model_centric.control_events import *
from .model_centric.fl_events import *
from .socket_handler import SocketHandler

# Websocket events routes
# This structure allows compatibility between javascript applications (syft.js/grid.js) and PyGrid.
routes = {
    CONTROL_EVENTS.SOCKET_PING: socket_ping,
    MODEL_CENTRIC_FL_EVENTS.HOST_FL_TRAINING: host_federated_training,
    MODEL_CENTRIC_FL_EVENTS.AUTHENTICATE: authenticate,
    MODEL_CENTRIC_FL_EVENTS.CYCLE_REQUEST: cycle_request,
    MODEL_CENTRIC_FL_EVENTS.REPORT: report,
    REQUEST_MSG.GET_ID: get_node_infos,
    REQUEST_MSG.CONNECT_NODE: connect_grid_nodes,
    REQUEST_MSG.HOST_MODEL: host_model,
    REQUEST_MSG.RUN_INFERENCE: run_inference,
    REQUEST_MSG.DELETE_MODEL: delete_model,
    REQUEST_MSG.LIST_MODELS: get_models,
    REQUEST_MSG.AUTHENTICATE: authentication,
}

handler = SocketHandler()


def route_requests(message, socket):
    """Handle a message from websocket connection and route them to the desired
    method.

    Args:
        message : message received.
    Returns:
        message_response : message response.
    """
    global routes

    if isinstance(message, bytearray):
        return forward_binary_message(message)

    try:
        message = json.loads(message)
        response = routes[message[REQUEST_MSG.TYPE_FIELD]](message)
        return response
    except Exception as e:
        return json.dumps({"error": str(e)})


@ws.route("/")
def socket_api(socket):
    """Handle websocket connections and receive their messages.

    Args:
        socket : websocket instance.
    """
    while not socket.closed:
        message = socket.receive()
        if not message:
            continue
        else:
            # Process received message
            response = route_requests(message, socket)
            if isinstance(response, bytearray):
                socket.send(response, binary=True)
            else:
                socket.send(response)

    worker_id = handler.remove(socket)
