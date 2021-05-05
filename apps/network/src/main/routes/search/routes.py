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
        route_logic, NetworkSearchMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )
