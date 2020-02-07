"""
This file exists to provide a route to websocket events.
"""
from .. import ws

from .webrtc_events import *
from .scope_events import *
from .control_events import *
from .socket_handler import SocketHandler

from ..codes import GRID_MSG

import json

# Websocket events routes
# This structure allows compatibility between javascript applications (syft.js/grid.js) and PyGrid.
routes = {
    GRID_MSG.SOCKET_PING: socket_ping,
    GRID_MSG.GET_PROTOCOL: get_protocol,
    GRID_MSG.JOIN_ROOM: scope_broadcast,
    GRID_MSG.INTERNAL_MSG: internal_message,
    GRID_MSG.PEER_LEFT: scope_broadcast,
}


handler = SocketHandler()


def route_requests(message, socket):
    """ Handle a message from websocket connection and route them to the desired method.

        Args:
            message : message received.
        Returns:
            message_response : message response.
    """
    global routes
    try:
        message = json.loads(message)
        return routes[message[GRID_MSG.TYPE_FIELD]](message, socket)
    except Exception as e:
        return json.dumps({"error": "Invalid JSON format/field!"})


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
            response = route_requests(message, socket)
            if isinstance(response, bytearray):
                socket.send(response, binary=True)
            else:
                socket.send(response)
    worker_id = handler.remove(socket)
