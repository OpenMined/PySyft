import json

from .. import ws
from ..codes import GRID_EVENTS
from .network import *
from .socket_handler import SocketHandler

socket_handler = SocketHandler()

routes = {
    GRID_EVENTS.JOIN: register_node,
    GRID_EVENTS.MONITOR_ANSWER: update_node,
    GRID_EVENTS.FORWARD: forward,
}


def route_request(message, socket):
    global routes

    message = json.loads(message)
    print("Message: ", message)
    if message and message.get(MSG_FIELD.TYPE, None) in routes.keys():
        return routes[message[MSG_FIELD.TYPE]](message, socket)
    else:
        return {"status": "error", "message": "Invalid request format!"}


@ws.route("/")
def socket_api(socket):
    while not socket.closed:
        message = socket.receive()
        if not message:
            continue
        else:
            response = route_request(message, socket)
            if response:
                socket.send(json.dumps(response))

    worker_id = socket_handler.remove(socket)
