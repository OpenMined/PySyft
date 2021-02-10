from .blueprint import users_blueprint as user_route
from flask import request, Response
import json

from syft.grid.messages.user_messages import (
    CreateUserMessage,
    DeleteUserMessage,
    GetUserMessage,
    GetUsersMessage,
    UpdateUserMessage,
    SearchUsersMessage,
)

from ..auth import error_handler, token_required, optional_token
from ...core.task_handler import route_logic, task_handler
from ...core.node import node
from ...core.exceptions import MissingRequestKeyError


@user_route.route("", methods=["POST"])
@optional_token
def create_user(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["current_user"] = current_user
    status_code, response_msg = error_handler(
        route_logic, CreateUserMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@user_route.route("/login", methods=["POST"])
def login_route():
    def route_logic():
        # Get request body
        content = json.loads(request.data)

        # Execute task
        response_body = task_handler(
            route_function=node.login,
            data=content,
            mandatory={
                "password": MissingRequestKeyError,
                "email": MissingRequestKeyError,
            },
        )
        return response_body

    status_code, response_body = error_handler(route_logic)

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@user_route.route("", methods=["GET"])
@token_required
def get_all_users_route(current_user):
    status_code, response_msg = error_handler(
        route_logic, GetUsersMessage, current_user, {}
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@user_route.route("/<user_id>", methods=["GET"])
@token_required
def get_specific_user_route(current_user, user_id):

    content = {}
    content["user_id"] = user_id

    status_code, response_msg = error_handler(
        route_logic, GetUserMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@user_route.route("/<user_id>/email", methods=["PUT"])
@token_required
def change_user_email_route(current_user, user_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["user_id"] = user_id

    status_code, response_msg = error_handler(
        route_logic, UpdateUserMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@user_route.route("/<user_id>/role", methods=["PUT"])
@token_required
def change_user_role_route(current_user, user_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["user_id"] = user_id

    status_code, response_msg = error_handler(
        route_logic, UpdateUserMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@user_route.route("/<user_id>/password", methods=["PUT"])
@token_required
def change_user_password_role(current_user, user_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["user_id"] = user_id

    status_code, response_msg = error_handler(
        route_logic, UpdateUserMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@user_route.route("/<user_id>/groups", methods=["PUT"])
@token_required
def change_user_groups_route(current_user, user_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["user_id"] = user_id

    status_code, response_msg = error_handler(
        route_logic, UpdateUserMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@user_route.route("/<user_id>", methods=["DELETE"])
@token_required
def delete_user_role(current_user, user_id):
    content = {}
    content["user_id"] = user_id

    status_code, response_msg = error_handler(
        route_logic, DeleteUserMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@user_route.route("/search", methods=["POST"])
@token_required
def search_users_route(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    status_code, response_msg = error_handler(
        route_logic, SearchUsersMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )
