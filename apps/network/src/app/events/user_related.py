import logging
from datetime import datetime, timedelta
from json import dumps, loads
from json.decoder import JSONDecodeError
from secrets import token_hex

import jwt
from ..codes import MSG_FIELD

from .. import db
from ..auth import error_handler, token_required_factory
from ..exceptions import (
    AuthorizationError,
    InvalidCredentialsError,
    MissingRequestKeyError,
    PyGridError,
    RoleNotFoundError,
    UserNotFoundError,
)
from ..database import Role, User
from ..database.utils import *
from ..users.user_ops import (
    change_user_email,
    change_user_password,
    change_user_role,
    delete_user,
    get_all_users,
    get_specific_user,
    login_user,
    search_users,
    signup_user,
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


def signup_user_socket(message: dict) -> str:
    def route_logic(message: dict) -> dict:
        private_key = user = user_role = None
        private_key = message.get("private-key")
        password = message.get("password")
        email = message.get("email")
        role = message.get("role")

        if email is None or password is None:
            raise MissingRequestKeyError

        user = signup_user(private_key, email, password, role)
        user = expand_user_object(user)

        response_body = {MSG_FIELD.SUCCESS: True, "user": user}
        return response_body

    status_code, response_body = error_handler(route_logic, message)

    return dumps(response_body)


def login_user_socket(message: dict) -> str:
    def route_logic(message: dict) -> dict:

        email = message.get("email")
        password = message.get("password")
        private_key = message.get("private-key")

        if email is None or password is None or private_key is None:
            raise MissingRequestKeyError

        token = login_user(private_key, email, password)
        response_body = {MSG_FIELD.SUCCESS: True, "token": token}
        return response_body

    status_code, response_body = error_handler(route_logic, message)

    return dumps(response_body)


@token_required
def get_all_users_socket(current_user: User, message: dict) -> str:
    def route_logic(current_user: User, message: dict) -> dict:
        private_key = message.get("private-key")
        if private_key is None:
            raise MissingRequestKeyError

        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        users = get_all_users(current_user, private_key)
        users = [expand_user_object(user) for user in users]
        response_body = {MSG_FIELD.SUCCESS: True, "users": users}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def get_specific_user_socket(current_user: User, message: dict) -> str:
    def route_logic(current_user: User, message: dict) -> dict:
        user_id = message.get("id")
        private_key = message.get("private-key")
        if private_key is None:
            raise MissingRequestKeyError

        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        user = get_specific_user(current_user, private_key, user_id)
        user = expand_user_object(user)
        response_body = {MSG_FIELD.SUCCESS: True, "user": user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def change_user_email_socket(current_user: User, message: dict) -> str:
    def route_logic(current_user: User, message: dict) -> dict:
        user_id = message.get("id")
        email = message.get("email")
        private_key = message.get("private-key")

        if email is None or private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        user = change_user_email(current_user, private_key, email, user_id)
        user = expand_user_object(user)
        response_body = {MSG_FIELD.SUCCESS: True, "user": user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def change_user_role_socket(current_user: User, message: dict) -> str:
    def route_logic(current_user: User, message: dict) -> dict:
        user_id = message.get("id")
        role = message.get("role")
        private_key = message.get("private-key")

        if role is None or private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        edited_user = change_user_role(current_user, private_key, role, user_id)
        edited_user = expand_user_object(edited_user)
        response_body = {MSG_FIELD.SUCCESS: True, "user": edited_user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def change_user_password_socket(current_user: User, message: dict) -> str:
    def route_logic(current_user: User, message: dict) -> dict:
        user_id = message.get("id")
        password = message.get("password")
        private_key = message.get("private-key")

        if password is None or private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        edited_user = change_user_password(current_user, private_key, password, user_id)
        edited_user = expand_user_object(edited_user)

        response_body = {MSG_FIELD.SUCCESS: True, "user": edited_user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def delete_user_socket(current_user: User, message: dict) -> str:
    def route_logic(current_user: User, message: dict) -> dict:
        user_id = message.get("id")
        private_key = message.get("private-key")

        if private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        edited_user = delete_user(current_user, private_key, user_id)
        edited_user = expand_user_object(edited_user)
        response_body = {MSG_FIELD.SUCCESS: True, "user": edited_user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)


@token_required
def search_users_socket(current_user: User, message: dict) -> str:
    def route_logic(current_user: User, message: dict) -> dict:
        filters = message.copy()
        filters.pop("private-key", None)
        filters.pop("token", None)

        email = message.get("email")
        role = message.get("role")

        private_key = message.get("private-key")

        if email is None and role is None:
            raise MissingRequestKeyError
        if private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        users = search_users(current_user, private_key, filters)
        users = [expand_user_object(user) for user in users]
        response_body = {MSG_FIELD.SUCCESS: True, "users": users}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, message)

    return dumps(response_body)
