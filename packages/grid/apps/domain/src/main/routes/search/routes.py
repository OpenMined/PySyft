# stdlib
import json

# third party
from flask import Response
from flask import request
from syft.grid.messages.network_search_message import NetworkSearchMessage

# grid relative
from ...core.task_handler import route_logic
from ..auth import error_handler
from ..auth import optional_token
from ..auth import token_required
from .blueprint import search_blueprint as search_route


@search_route.route("/", methods=["GET"])
@optional_token
def broadcast_search(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    status_code, response_msg = error_handler(
        route_logic, 200, NetworkSearchMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@search_route.route("/domain-search", methods=["POST"])
@optional_token
def domain_search(current_user):
    # TODO: This route must be refactored in the future to follow the same
    # pattern adopted by PyGrid services.

    from ...core.node import get_node  # TODO: fix circular import

    associations = get_node().association_requests.associations()

    # Get request body
    content = request.get_json()
    if not content:
        return Response(
            json.dumps({"error": "Invalid message body!"}),
            status=400,
            mimetype="application/json",
        )

    queries = set(content.get("query", []))

    _count = 0
    # AND search
    for dataset in get_node().store.values():
        if queries.issubset(set(dataset.tags)):
            _count += 1

    result = {"node": get_node().name, "items": _count}

    return Response(
        json.dumps(result),
        status=200,
        mimetype="application/json",
    )
