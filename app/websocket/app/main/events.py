"""
This file exists to provide one common place for all websocket events.
"""

from . import hook, local_worker, ws

from .event_routes import *

from .persistence.utils import recover_objects, snapshot

import json

routes = {
    "get-id": get_node_id,
    "connect-node": connect_grid_nodes,
    "syft-command": syft_command,
    "socket-ping": socket_ping,
    "host-model": host_model,
    "run-inference": run_inference,
    "delete-model": delete_model,
    "list-models": get_models,
    "download-model": download_model,
}


def route_requests(message):
    global routes
    try:
        message = json.loads(message)
        return routes[message["type"]](message)
    except Exception as e:
        print("Exception: ", e)
        return json.dumps({"error": "Invalid JSON format/field!"})


@ws.route("/")
def socket_api(socket):
    while not socket.closed:
        message = socket.receive()
        if not message:
            continue
        else:
            if isinstance(message, bytearray):
                # Forward syft commands to syft worker

                # Load previous database tensors
                if not local_worker._objects:
                    recover_objects(local_worker)

                decoded_response = local_worker._recv_msg(message)

                # Save local worker state at database
                snapshot(local_worker)

                socket.send(decoded_response, binary=True)
            else:
                socket.send(route_requests(message))
