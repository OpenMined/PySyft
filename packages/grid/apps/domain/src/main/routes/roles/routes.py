# stdlib
import json

# third party
from flask import Response
from flask import request
from syft.grid.messages.role_messages import CreateRoleMessage
from syft.grid.messages.role_messages import DeleteRoleMessage
from syft.grid.messages.role_messages import GetRoleMessage
from syft.grid.messages.role_messages import GetRolesMessage
from syft.grid.messages.role_messages import UpdateRoleMessage

# grid relative
from ...core.task_handler import route_logic
from ..auth import error_handler
from ..auth import token_required
from .blueprint import roles_blueprint as roles_route


@roles_route.route("", methods=["POST"])
@token_required
def create_role_route(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    status_code, response_msg = error_handler(
        route_logic, 204, CreateRoleMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@roles_route.route("/<role_id>", methods=["GET"])
@token_required
def get_role_route(current_user, role_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["role_id"] = role_id

    status_code, response_msg = error_handler(
        route_logic, 200, GetRoleMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@roles_route.route("", methods=["GET"])
@token_required
def get_all_roles_route(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    status_code, response_msg = error_handler(
        route_logic, 200, GetRolesMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@roles_route.route("/<role_id>", methods=["PUT"])
@token_required
def put_role_route(current_user, role_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["role_id"] = role_id

    status_code, response_msg = error_handler(
        route_logic, 204, UpdateRoleMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@roles_route.route("/<role_id>", methods=["DELETE"])
@token_required
def delete_role_route(current_user, role_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["role_id"] = role_id

    status_code, response_msg = error_handler(
        route_logic, 204, DeleteRoleMessage, current_user, content
    )
    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )
