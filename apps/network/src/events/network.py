import json
import threading
import time

from ..codes import MSG_FIELD
from .socket_handler import SocketHandler

socket_handler = SocketHandler()


def update_node(message, socket):
    """Update a node in the socket handler.

    Args:
        message (dict): The message containing the ID of the node to be updated.
        socket: Socket descriptor of the node to be updated.
    """
    try:
        worker = socket_handler.get(socket)
        worker.update_node_infos(message)
    except Exception:
        pass


def register_node(message, socket):
    """Register a new node to the socket handler.

    Args:
        message (dict): The message containing the ID of the node to be registered.
        socket: Socket descriptor of the node to be registered.

    Returns:
        dict or None: Returns a status of success if the node is registered.
    """
    try:
        time.sleep(1)
        node_id = message[MSG_FIELD.NODE_ID]
        worker = socket_handler.new_connection(node_id, socket)
        t = threading.Thread(target=worker.monitor)
        t.start()
        return {"status": "success!"}
    except Exception:
        pass


def forward(message, socket):
    """Forward message to a specific node.

    Args:
        message (dict): The message which contains the ID of the node to be forwarded to.
        socket: Socket descriptor of the node to which the message is forwarded.
    """
    try:
        time.sleep(1)
        dest = message[MSG_FIELD.DESTINATION]
        worker = socket_handler.get(dest)
        if worker:
            content = message[MSG_FIELD.CONTENT]
            worker.send(json.dumps(content))
    except Exception:
        pass
