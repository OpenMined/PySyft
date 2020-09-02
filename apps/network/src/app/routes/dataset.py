import json

from flask import Response

from .. import http
from ..events import SocketHandler

socket_handler = SocketHandler()


@http.route("/datasets", methods=["GET"])
def get_datasets():
    """Return a response object with all the nodes in the network and their
    hosted datasets."""
    try:
        response = {}

        for node_id, node in socket_handler.nodes:
            response[node_id] = node.hosted_datasets

        response_body = json.dumps(response)
        return Response(response_body, status=200, mimetype="application/json")
    except Exception:
        return Response({}, status=500, mimetype="application/json")
