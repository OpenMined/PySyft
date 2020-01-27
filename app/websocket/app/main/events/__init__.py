"""
This file exists to provide a route to websocket events.
"""

from .. import hook, local_worker, ws

from .syft_events import *
from .model_events import *
from .control_events import *

from syft.codes import REQUEST_MSG
import json

# Websocket events routes
# This structure allows compatibility between javascript applications (syft.js/grid.js) and PyGrid.
routes = {
    REQUEST_MSG.GET_ID: get_node_id,
    REQUEST_MSG.CONNECT_NODE: connect_grid_nodes,
    REQUEST_MSG.HOST_MODEL: host_model,
    REQUEST_MSG.RUN_INFERENCE: run_inference,
    REQUEST_MSG.DELETE_MODEL: delete_model,
    REQUEST_MSG.LIST_MODELS: get_models,
    REQUEST_MSG.AUTHENTICATE: authentication,
}


def route_requests(message):
    """ Handle a message from websocket connection and route them to the desired method.

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
        return routes[message[REQUEST_MSG.TYPE_FIELD]](message)
    except Exception as e:
        return json.dumps({"error": str(e)})


@ws.route("/")
def socket_api(socket):
    """ Handle websocket connections and receive their messages.
    
        Args:
            socket : websocket instance.
    """
    while not socket.closed:
        message = socket.receive()
        if not message:
            continue
        else:
            response = route_requests(message)
            if isinstance(response, bytearray):
                socket.send(response, binary=True)
            else:
                socket.send(response)
