import json

from flask import Response

from .. import http
from ..events import SocketHandler

socket_handler = SocketHandler()


@http.route("/models", methods=["GET"])
def get_models():
    """Return a response object containing all the models on all the nodes in
    the network."""
    try:
        response = {}
        models = []
        for node_id, node in socket_handler.nodes:
            models.append(node.hosted_models)

        response["models"] = models
        response_body = json.dumps(response)
        return Response(response_body, status=200, mimetype="application/json")
    except Exception:
        return Response({}, status=500, mimetype="application/json")
