"""This file exists to provide a route to websocket events."""
# stdlib
import json

# grid relative
from .. import ws
from ..core.codes import *
from ..core.codes import GROUP_EVENTS
from ..core.codes import ROLE_EVENTS
from ..core.codes import USER_EVENTS
from .model_centric.fl_events import *
from .model_centric.socket_handler import SocketHandler


class REQUEST_MSG(object):  # noqa: N801
    TYPE_FIELD = "type"
    GET_ID = "get-id"
    CONNECT_NODE = "connect-node"
    HOST_MODEL = "host-model"
    RUN_INFERENCE = "run-inference"
    LIST_MODELS = "list-models"
    DELETE_MODEL = "delete-model"
    RUN_INFERENCE = "run-inference"
    AUTHENTICATE = "authentication"


# Websocket events routes
# This structure allows compatibility between javascript applications (syft.js/grid.js) and PyGrid.
routes = {
    MODEL_CENTRIC_FL_EVENTS.HOST_FL_TRAINING: host_federated_training,
    MODEL_CENTRIC_FL_EVENTS.AUTHENTICATE: authenticate,
    MODEL_CENTRIC_FL_EVENTS.CYCLE_REQUEST: cycle_request,
    MODEL_CENTRIC_FL_EVENTS.REPORT: report,
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

    request_id = None
    try:
        message = json.loads(message)
        request_id = message.get(MSG_FIELD.REQUEST_ID)
        response = routes[message[REQUEST_MSG.TYPE_FIELD]](message)
    except Exception as e:
        response = {"error": str(e)}

    if request_id:
        response[MSG_FIELD.REQUEST_ID] = request_id

    return json.dumps(response)


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
