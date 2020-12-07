from json import dumps, loads
from json.decoder import JSONDecodeError
import logging

from flask import Response, request

from ...core.codes import RESPONSE_MSG
from ...core.exceptions import (
    UserNotFoundError,
    RoleNotFoundError,
    InvalidCredentialsError,
    AuthorizationError,
    PyGridError,
    MissingRequestKeyError,
)
from .blueprint import roles_blueprint as roles_route
from ...core.task_handler import task_handler
from ..auth import error_handler, token_required_factory
from ...core.roles.role_ops import (
    create_role,
    get_role,
    get_all_roles,
    put_role,
    delete_role,
)
from ...core.database import db, Role, User, model_to_json


def get_token(*args, **kwargs):
    token = request.headers.get("token")
    if token is None:
        raise MissingRequestKeyError

    return token


def format_result(response_body, status_code, mimetype):
    return Response(dumps(response_body), status=status_code, mimetype=mimetype)


token_required = token_required_factory(get_token, format_result)


@roles_route.route("", methods=["POST"])
@token_required
def create_role_route(current_user):
    def route_logic():
        content = loads(request.data)
        content = {"role_fields": content}
        content["current_user"] = current_user

        response_body = task_handler(
            route_function=create_role,
            data=content,
            mandatory={
                "current_user": MissingRequestKeyError,
                "role_fields": MissingRequestKeyError,
            },
        )
        return response_body

    status_code, response_body = error_handler(route_logic)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@roles_route.route("/<role_id>", methods=["GET"])
@token_required
def get_role_route(current_user, role_id):
    def route_logic(current_user, role_id):
        content = {"current_user": current_user, "role_id": role_id}

        response_body = task_handler(
            route_function=get_role,
            data=content,
            mandatory={
                "current_user": MissingRequestKeyError,
                "role_id": MissingRequestKeyError,
            },
        )

        return response_body

    role_id = int(role_id)
    status_code, response_body = error_handler(route_logic, current_user, role_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@roles_route.route("", methods=["GET"])
@token_required
def get_all_roles_route(current_user):
    def route_logic(current_user):
        content = {"current_user": current_user}
        response_body = task_handler(
            route_function=get_all_roles,
            data=content,
            mandatory={
                "current_user": MissingRequestKeyError,
            },
        )

        return response_body

    status_code, response_body = error_handler(route_logic, current_user)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@roles_route.route("/<role_id>", methods=["PUT"])
@token_required
def put_role_route(current_user, role_id):
    def route_logic(current_user, role_id):
        new_fields = loads(request.data)
        content = {
            "current_user": current_user,
            "role_id": role_id,
            "new_fields": new_fields,
        }
        response_body = task_handler(
            route_function=put_role,
            data=content,
            mandatory={
                "current_user": MissingRequestKeyError,
                "role_id": MissingRequestKeyError,
                "new_fields": MissingRequestKeyError,
            },
        )

        return response_body

    status_code, response_body = error_handler(route_logic, current_user, role_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@roles_route.route("/<role_id>", methods=["DELETE"])
@token_required
def delete_role_route(current_user, role_id):
    def route_logic(current_user, role_id):
        content = {
            "current_user": current_user,
            "role_id": role_id,
        }
        response_body = task_handler(
            route_function=delete_role,
            data=content,
            mandatory={
                "current_user": MissingRequestKeyError,
                "role_id": MissingRequestKeyError,
            },
        )

        return response_body

    status_code, response_body = error_handler(route_logic, current_user, role_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )
