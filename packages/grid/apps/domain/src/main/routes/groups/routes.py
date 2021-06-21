# stdlib
import json

# third party
from flask import Response
from flask import request
from syft.grid.messages.group_messages import CreateGroupMessage
from syft.grid.messages.group_messages import DeleteGroupMessage
from syft.grid.messages.group_messages import GetGroupMessage
from syft.grid.messages.group_messages import GetGroupsMessage
from syft.grid.messages.group_messages import UpdateGroupMessage

# grid relative
from ...core.task_handler import route_logic
from ..auth import error_handler
from ..auth import token_required
from .blueprint import groups_blueprint as group_route


@group_route.route("", methods=["POST"])
@token_required
def create_group_route(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["current_user"] = current_user

    status_code, response_msg = error_handler(
        route_logic, 200, CreateGroupMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@group_route.route("", methods=["GET"])
@token_required
def get_all_groups_routes(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["current_user"] = current_user

    status_code, response_msg = error_handler(
        route_logic, 200, GetGroupsMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@group_route.route("/<group_id>", methods=["GET"])
@token_required
def get_specific_group_route(current_user, group_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["current_user"] = current_user
    content["group_id"] = group_id

    status_code, response_msg = error_handler(
        route_logic, 200, GetGroupMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@group_route.route("/<group_id>", methods=["PUT"])
@token_required
def update_group_route(current_user, group_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["current_user"] = current_user
    content["group_id"] = group_id

    status_code, response_msg = error_handler(
        route_logic, 204, UpdateGroupMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@group_route.route("/<group_id>", methods=["DELETE"])
@token_required
def delete_group_route(current_user, group_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["current_user"] = current_user
    content["group_id"] = group_id

    status_code, response_msg = error_handler(
        route_logic, 204, DeleteGroupMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )
