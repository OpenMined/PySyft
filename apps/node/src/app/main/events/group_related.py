from datetime import datetime, timedelta
from json import dumps, loads
from json.decoder import JSONDecodeError
from secrets import token_hex

import jwt
from bcrypt import checkpw, gensalt, hashpw
from flask import Response
from flask import current_app as app
from flask import request
from syft.codes import RESPONSE_MSG
from werkzeug.security import check_password_hash, generate_password_hash

from ... import db
from ..auth import error_handler, token_required_factory
from ..core.exceptions import (
    AuthorizationError,
    GroupNotFoundError,
    InvalidCredentialsError,
    MissingRequestKeyError,
    PyGridError,
    RoleNotFoundError,
    UserNotFoundError,
)
from ..database import Group, Role, User, UserGroup
from ..database.utils import *
from ..users.group_ops import (
    create_group,
    get_group,
    get_all_groups,
    put_group,
    delete_group,
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
def create_group_socket(current_user: User, message: dict) -> str:
    def route_logic(current_user: User, message: dict) -> dict:
        name = message.get("name")
        private_key = message.get("private-key")

        if name is None or private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        group = create_group(current_user, private_key, name)
        group = model_to_json(group)
        response_body = {RESPONSE_MSG.SUCCESS: True, "group": group}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def get_group_socket(current_user: User, message: dict) -> str:
    def route_logic(current_user: User, message: dict) -> dict:
        group_id = message.get("id")
        private_key = message.get("private-key")

        if private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        group = get_group(current_user, private_key, group_id)
        group = model_to_json(group)
        response_body = {RESPONSE_MSG.SUCCESS: True, "group": group}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def get_all_groups_socket(current_user: User, message: dict) -> str:
    def route_logic(current_user: User, message: dict) -> dict:
        private_key = message.get("private-key")

        if private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        groups = get_all_groups(current_user, private_key)
        groups = [model_to_json(group) for group in groups]
        response_body = {RESPONSE_MSG.SUCCESS: True, "groups": groups}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def put_group_socket(current_user: User, message: dict) -> str:
    def route_logic(current_user: User, message: dict) -> dict:
        private_key = message.get("private-key")
        new_group = message.get("group")
        group_id = message.get("id")

        if new_group is None or private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        group = put_group(current_user, private_key, group_id, new_group)
        group = model_to_json(group)
        response_body = {RESPONSE_MSG.SUCCESS: True, "group": group}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def delete_group_socket(current_user: User, message: dict) -> str:
    def route_logic(current_user: User, message: dict) -> dict:
        group_id = message.get("id")
        private_key = message.get("private-key")

        if private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        group = delete_group(current_user, private_key, group_id)
        group = model_to_json(group)
        response_body = {RESPONSE_MSG.SUCCESS: True, "group": group}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)
