from json import dumps
from json.decoder import JSONDecodeError
import logging

from ..codes import MSG_FIELD
from ..exceptions import (
    UserNotFoundError,
    RoleNotFoundError,
    InvalidCredentialsError,
    AuthorizationError,
    PyGridError,
    MissingRequestKeyError,
)
from ..auth import error_handler, token_required_factory
from ..users.role_ops import create_role, get_role, get_all_roles, put_role, delete_role
from ..database import Role, User
from ..database.utils import model_to_json
from .. import db

expected_fields = (
    "name",
    "can_edit_settings",
    "can_create_users",
    "can_edit_roles",
    "can_manage_nodes",
)


def get_token(*args, **kwargs):
    message = args[0]
    token = message.get("token")
    if token is None:
        raise MissingRequestKeyError

    return token


def format_result(response_body, status_code, mimetype):
    return dumps(response_body)


token_required = token_required_factory(get_token, format_result)


@token_required
def create_role_socket(current_user, message: dict) -> str:
    def route_logic(current_user, message: dict) -> dict:
        private_key = message.get("private-key")
        role = message.get("role")

        if private_key is None or role is None:
            raise MissingRequestKeyError

        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        for field in expected_fields:
            if role.get(field) is None:
                raise MissingRequestKeyError

        new_role = create_role(current_user, private_key, role)
        new_role = model_to_json(new_role)
        response_body = {MSG_FIELD.SUCCESS: True, "role": new_role}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def get_role_socket(current_user, message: dict) -> str:
    def route_logic(current_user, message: dict) -> dict:
        private_key = message.get("private-key")
        role_id = message.get("id")

        if private_key is None or role_id is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        role = get_role(current_user, private_key, role_id)
        role = model_to_json(role)
        response_body = {MSG_FIELD.SUCCESS: True, "role": role}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def get_all_roles_socket(current_user, message: dict) -> str:
    def route_logic(current_user, message: dict) -> dict:
        private_key = message.get("private-key")

        if private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        roles = get_all_roles(current_user, private_key)
        roles = [model_to_json(r) for r in roles]
        response_body = {MSG_FIELD.SUCCESS: True, "roles": roles}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def put_role_socket(current_user, message: dict) -> str:
    def route_logic(current_user, message: dict) -> dict:
        private_key = message.get("private-key")
        role_id = message.get("id")
        role = message.get("role")

        if private_key is None or role_id is None or role is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        role = put_role(current_user, role_id, role)
        role = model_to_json(role)
        response_body = {MSG_FIELD.SUCCESS: True, "role": role}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def delete_role_socket(current_user, message: dict) -> str:
    def route_logic(current_user, message: dict) -> dict:
        private_key = message.get("private-key")
        role_id = message.get("id")

        if private_key is None or role_id is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        deleted_user = delete_role(current_user, role_id)
        deleted_user = model_to_json(deleted_user)
        response_body = {MSG_FIELD.SUCCESS: True, "user": deleted_user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)
