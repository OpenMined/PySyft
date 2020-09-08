from secrets import token_hex
from json import dumps, loads
from json.decoder import JSONDecodeError
from datetime import datetime, timedelta

import jwt
from bcrypt import hashpw, checkpw, gensalt
from syft.codes import RESPONSE_MSG
from flask import request, Response
from flask import current_app as app
from werkzeug.security import generate_password_hash, check_password_hash

from ..core.exceptions import (
    PyGridError,
    UserNotFoundError,
    RoleNotFoundError,
    GroupNotFoundError,
    AuthorizationError,
    MissingRequestKeyError,
    InvalidCredentialsError,
)
from ... import db
from .. import main_routes
from ..database import Role, User, UserGroup, Group
from ..database.utils import *
from ..users.user_ops import (
    signup_user,
    login_user,
    get_all_users,
    get_specific_user,
    change_usr_email,
    change_usr_role,
    change_usr_password,
    change_usr_groups,
    delete_user,
    search_users,
)
from ..auth import token_required_factory, error_handler


def get_token(*args, **kwargs):
    token = request.headers.get("token")
    if token is None:
        raise MissingRequestKeyError

    return token


def format_result(response_body, status_code, mimetype):
    return Response(dumps(response_body), status=status_code, mimetype=mimetype)


token_required = token_required_factory(get_token, format_result)


@main_routes.route("/users", methods=["POST"])
def signup_user_route():
    def route_logic():
        private_key = usr = usr_role = None
        private_key = request.headers.get("private-key")
        data = loads(request.data)
        password = data.get("password")
        email = data.get("email")
        role = data.get("role")

        if email is None or password is None:
            raise MissingRequestKeyError

        user = signup_user(private_key, email, password, role)
        user = expand_user_object(user)
        response_body = {RESPONSE_MSG.SUCCESS: True, "user": user}
        return response_body

    status_code, response_body = error_handler(route_logic)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/users/login", methods=["POST"])
def login_user_route():
    def route_logic():
        data = loads(request.data)
        email = data.get("email")
        password = data.get("password")
        private_key = request.headers.get("private-key")

        if email is None or password is None or private_key is None:
            raise MissingRequestKeyError

        token = login_user(private_key, email, password)
        response_body = {RESPONSE_MSG.SUCCESS: True, "token": token}
        return response_body

    status_code, response_body = error_handler(route_logic)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/users", methods=["GET"])
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
        response_body = {RESPONSE_MSG.SUCCESS: True, "users": users}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/users/<user_id>", methods=["GET"])
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
        response_body = {RESPONSE_MSG.SUCCESS: True, "user": user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, user_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/users/<user_id>/email", methods=["PUT"])
@token_required
def change_usr_email_route(current_user, user_id):
    def route_logic(current_user, user_id):
        user_id = int(user_id)
        data = loads(request.data)
        email = data.get("email")
        private_key = request.headers.get("private-key")

        if email is None or private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        user = change_usr_email(current_user, private_key, email, user_id)
        user = expand_user_object(user)
        response_body = {RESPONSE_MSG.SUCCESS: True, "user": user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, user_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/users/<user_id>/role", methods=["PUT"])
@token_required
def change_usr_role_route(current_user, user_id):
    def route_logic(current_user, user_id):
        user_id = int(user_id)
        data = loads(request.data)
        role = data.get("role")
        private_key = request.headers.get("private-key")

        if role is None or private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        edited_user = change_usr_role(current_user, private_key, role, user_id)
        edited_user = expand_user_object(edited_user)
        response_body = {RESPONSE_MSG.SUCCESS: True, "user": edited_user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, user_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/users/<user_id>/password", methods=["PUT"])
@token_required
def change_usr_password_role(current_user, user_id):
    def route_logic(current_user, user_id):
        user_id = int(user_id)
        data = loads(request.data)
        password = data.get("password")
        private_key = request.headers.get("private-key")

        if password is None or private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        edited_user = change_usr_password(current_user, private_key, password, user_id)
        edited_user = expand_user_object(edited_user)

        response_body = {RESPONSE_MSG.SUCCESS: True, "user": edited_user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, user_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/users/<user_id>/groups", methods=["PUT"])
@token_required
def change_usr_groups_route(current_user, user_id):
    def route_logic(current_user, user_id):
        user_id = int(user_id)
        data = loads(request.data)
        groups = data.get("groups")
        private_key = request.headers.get("private-key")

        if groups is None or private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        edited_user = change_usr_groups(current_user, private_key, groups, user_id)
        edited_user = expand_user_object(edited_user)
        response_body = {RESPONSE_MSG.SUCCESS: True, "user": edited_user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, user_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/users/<user_id>", methods=["DELETE"])
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
        response_body = {RESPONSE_MSG.SUCCESS: True, "user": edited_user}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, user_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@main_routes.route("/users/search", methods=["POST"])
@token_required
def search_users_route(current_user):
    def route_logic(current_user):
        filters = loads(request.data)
        email = filters.get("email")
        role = filters.get("role")
        group = filters.get("group")

        private_key = request.headers.get("private-key")

        if email is None and role is None and group is None:
            raise MissingRequestKeyError
        if private_key is None:
            raise MissingRequestKeyError
        if private_key != current_user.private_key:
            raise InvalidCredentialsError

        users = search_users(current_user, private_key, filters, group)
        users = [expand_user_object(user) for user in users]
        response_body = {RESPONSE_MSG.SUCCESS: True, "users": users}
        return response_body

    status_code, response_body = error_handler(route_logic, current_user)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )
