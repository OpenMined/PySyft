from json import dumps, loads
from json.decoder import JSONDecodeError
import logging

from flask import Response, request
from syft.codes import RESPONSE_MSG

from ..core.exceptions import (
    UserNotFoundError,
    RoleNotFoundError,
    InvalidCredentialsError,
    AuthorizationError,
    PyGridError,
    MissingRequestKeyError,
)
from .. import main_routes
from ..auth import error_handler, token_required_factory
from ..users.role_ops import create_role, get_role, get_all_roles, put_role, delete_role
from ..database import Role, User
from ..database.utils import model_to_json
from ... import BaseModel, db

expected_fields = (
    "name",
    "can_triage_requests",
    "can_edit_settings",
    "can_create_users",
    "can_create_groups",
    "can_edit_roles",
    "can_manage_infrastructure",
    "can_upload_data",
)


def get_token(*args, **kwargs):
    token = request.headers.get("token")
    if token is None:
        raise MissingRequestKeyError

    return token


def format_result(response_body, status_code, mimetype):
    return Response(dumps(response_body), status=status_code, mimetype=mimetype)


token_required = token_required_factory(get_token, format_result)


@main_routes.route("/roles", methods=["POST"])
@token_required
def create_role_route(current_user):
    def route_logic(current_user):
        private_key = request.headers.get("private-key")
        data = loads(request.data)

        if private_key is None:
            raise MissingRequestKeyError

        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        for field in expected_fields:
            if data.get(field) is None:
                raise MissingRequestKeyError

        new_role = create_role(current_user, private_key, data)
        new_role = model_to_json(new_role)
        response_body = {RESPONSE_MSG.SUCCESS: True, "role": new_role}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/roles/<role_id>", methods=["GET"])
@token_required
def get_role_route(current_user, role_id):
    def route_logic(current_user, role_id):
        private_key = request.headers.get("private-key")

        if private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        role = get_role(current_user, private_key, role_id)
        role = model_to_json(role)
        response_body = {RESPONSE_MSG.SUCCESS: True, "role": role}
        return response_body

    role_id = int(role_id)
    status_code, response_body = error_handler(route_logic, current_user, role_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/roles", methods=["GET"])
@token_required
def get_all_roles_route(current_user):
    def route_logic(current_user):
        private_key = request.headers.get("private-key")

        if private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        roles = get_all_roles(current_user, private_key)
        roles = [model_to_json(r) for r in roles]
        response_body = {RESPONSE_MSG.SUCCESS: True, "roles": roles}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/roles/<role_id>", methods=["PUT"])
@token_required
def put_role_route(current_user, role_id):
    def route_logic(current_user, role_id):
        private_key = request.headers.get("private-key")
        new_fields = loads(request.data)

        if private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        role = put_role(current_user, role_id, new_fields)
        role = model_to_json(role)
        response_body = {RESPONSE_MSG.SUCCESS: True, "role": role}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, role_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/roles/<role_id>", methods=["DELETE"])
@token_required
def delete_role_route(current_user, role_id):
    def route_logic(current_user, role_id):
        private_key = request.headers.get("private-key")

        if private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        deleted_user = delete_role(current_user, role_id)
        deleted_user = model_to_json(deleted_user)
        response_body = {RESPONSE_MSG.SUCCESS: True, "user": deleted_user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, role_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )
