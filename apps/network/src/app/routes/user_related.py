from datetime import datetime, timedelta
from json import dumps, loads
from json.decoder import JSONDecodeError
from secrets import token_hex

import jwt
from bcrypt import checkpw, gensalt, hashpw
from flask import Response
from flask import current_app as app
from flask import request
from ..codes import MSG_FIELD

from .. import db
from .. import http
from ..auth import error_handler, token_required_factory
from ..exceptions import (
    AuthorizationError,
    InvalidCredentialsError,
    MissingRequestKeyError,
    PyGridError,
    RoleNotFoundError,
    UserNotFoundError,
)
from ..database import Role, User, create_role
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
    token = request.headers.get("token")
    if token is None:
        raise MissingRequestKeyError

    return token


def format_result(response_body, status_code, mimetype):
    return Response(dumps(response_body), status=status_code, mimetype=mimetype)


token_required = token_required_factory(get_token, format_result)


@http.route("/users", methods=["POST"])
def signup_user_route():
    def route_logic():
        private_key = user = user_role = None
        private_key = request.headers.get("private-key")
        data = loads(request.data)
        password = data.get("password")
        email = data.get("email")
        role = data.get("role")

        if email is None or password is None:
            raise MissingRequestKeyError

        user = signup_user(private_key, email, password, role)
        user = expand_user_object(user)
        response_body = {MSG_FIELD.SUCCESS: True, "user": user}
        return response_body

    status_code, response_body = error_handler(route_logic)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/users/login", methods=["POST"])
def login_user_route():
    def route_logic():
        data = loads(request.data)
        email = data.get("email")
        password = data.get("password")
        private_key = request.headers.get("private-key")

        if email is None or password is None or private_key is None:
            raise MissingRequestKeyError

        token = login_user(private_key, email, password)
        response_body = {MSG_FIELD.SUCCESS: True, "token": token}
        return response_body

    status_code, response_body = error_handler(route_logic)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/users", methods=["GET"])
@token_required
def get_all_users_route(current_user):
    def route_logic(current_user):
        private_key = request.headers.get("private-key")
        if private_key is None:
            raise MissingRequestKeyError

        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        users = get_all_users(current_user, private_key)
        users = [expand_user_object(user) for user in users]
        response_body = {MSG_FIELD.SUCCESS: True, "users": users}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/users/<user_id>", methods=["GET"])
@token_required
def get_specific_user_route(current_user, user_id):
    def route_logic(current_user, user_id):
        user_id = int(user_id)
        private_key = request.headers.get("private-key")
        if private_key is None:
            raise MissingRequestKeyError

        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        user = get_specific_user(current_user, private_key, user_id)
        user = expand_user_object(user)
        response_body = {MSG_FIELD.SUCCESS: True, "user": user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, user_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/users/<user_id>/email", methods=["PUT"])
@token_required
def change_user_email_route(current_user, user_id):
    def route_logic(current_user, user_id):
        user_id = int(user_id)
        data = loads(request.data)
        email = data.get("email")
        private_key = request.headers.get("private-key")

        if email is None or private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        user = change_user_email(current_user, private_key, email, user_id)
        user = expand_user_object(user)
        response_body = {MSG_FIELD.SUCCESS: True, "user": user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, user_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/users/<user_id>/role", methods=["PUT"])
@token_required
def change_user_role_route(current_user, user_id):
    def route_logic(current_user, user_id):
        user_id = int(user_id)
        data = loads(request.data)
        role = data.get("role")
        private_key = request.headers.get("private-key")

        if role is None or private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        edited_user = change_user_role(current_user, private_key, role, user_id)
        edited_user = expand_user_object(edited_user)
        response_body = {MSG_FIELD.SUCCESS: True, "user": edited_user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, user_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/users/<user_id>/password", methods=["PUT"])
@token_required
def change_user_password_role(current_user, user_id):
    def route_logic(current_user, user_id):
        user_id = int(user_id)
        data = loads(request.data)
        password = data.get("password")
        private_key = request.headers.get("private-key")

        if password is None or private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        edited_user = change_user_password(current_user, private_key, password, user_id)
        edited_user = expand_user_object(edited_user)

        response_body = {MSG_FIELD.SUCCESS: True, "user": edited_user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, user_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/users/<user_id>", methods=["DELETE"])
@token_required
def delete_user_role(current_user, user_id):
    def route_logic(current_user, user_id):
        user_id = int(user_id)
        private_key = request.headers.get("private-key")

        if private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        edited_user = delete_user(current_user, private_key, user_id)
        edited_user = expand_user_object(edited_user)
        response_body = {MSG_FIELD.SUCCESS: True, "user": edited_user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, user_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/users/search", methods=["POST"])
@token_required
def search_users_route(current_user):
    def route_logic(current_user):
        filters = loads(request.data)
        email = filters.get("email")
        role = filters.get("role")

        private_key = request.headers.get("private-key")

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

    status_code, response_body = error_handler(route_logic, current_user)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )
