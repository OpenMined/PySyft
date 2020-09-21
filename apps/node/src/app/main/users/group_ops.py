import logging
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
from .. import main_routes
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


def identify_user(private_key):
    if private_key is None:
        raise MissingRequestKeyError

    user = db.session.query(User).filter_by(private_key=private_key).one_or_none()
    if user is None:
        raise UserNotFoundError

    user_role = db.session.query(Role).get(user.role)
    if user_role is None:
        raise RoleNotFoundError

    return user, user_role


def create_group(current_user, private_key, name):
    user_role = Role.query.get(current_user.role)
    if user_role is None:
        raise RoleNotFoundError

    if not user_role.can_create_groups:
        raise AuthorizationError

    new_group = Group(name=name)
    db.session.add(new_group)
    db.session.commit()
    return new_group


def get_group(current_user, private_key, group_id):
    user_role = Role.query.get(current_user.role)
    if user_role is None:
        raise RoleNotFoundError

    if not user_role.can_triage_requests:
        raise AuthorizationError

    group = Group.query.get(group_id)
    if group is None:
        raise GroupNotFoundError

    return group


def get_all_groups(current_user, private_key):
    user_role = Role.query.get(current_user.role)
    if user_role is None:
        raise RoleNotFoundError

    if not user_role.can_triage_requests:
        raise AuthorizationError

    groups = Group.query.all()

    return groups


def put_group(current_user, private_key, group_id, new_fields):
    user_role = db.session.query(Role).get(current_user.role)

    if user_role is None:
        raise RoleNotFoundError
    if not user_role.can_create_groups:
        raise AuthorizationError

    group = db.session.query(Group).get(group_id)
    if group is None:
        raise GroupNotFoundError

    for key, value in new_fields.items():
        setattr(group, key, value)

    db.session.commit()
    return group


def delete_group(current_user, private_key, group_id):
    user_role = db.session.query(Role).get(current_user.role)

    if user_role is None:
        raise RoleNotFoundError
    if not user_role.can_create_groups:
        raise AuthorizationError

    group = db.session.query(Group).get(group_id)
    if group is None:
        raise GroupNotFoundError

    db.session.delete(group)
    db.session.commit()

    return group
