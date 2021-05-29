# stdlib
import json

# third party
from flask import Response
from flask import request
from syft.grid.messages.setup_messages import CreateInitialSetUpMessage
from syft.grid.messages.setup_messages import GetSetUpMessage

# grid relative
from ...core.task_handler import route_logic
from ..auth import error_handler
from ..auth import optional_token
from ..auth import token_required
from .blueprint import setup_blueprint as setup_route


@setup_route.route("", methods=["POST"])
@optional_token
def initial_setup(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    status_code, response_msg = error_handler(
        route_logic, 200, CreateInitialSetUpMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@setup_route.route("", methods=["GET"])
@token_required
def get_setup(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    status_code, response_msg = error_handler(
        route_logic, 200, GetSetUpMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@setup_route.route("/status", methods=["GET"])
def get_status():
    # third party
    from main.core.node import get_node  # TODO: fix circular import

    response = {"node_name": get_node().name, "init": len(get_node().users) > 0}

    return Response(json.dumps(response), status=200, mimetype="application/json")
