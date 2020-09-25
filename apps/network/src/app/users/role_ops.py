from json import dumps, loads
import logging

from ..exceptions import (
    UserNotFoundError,
    RoleNotFoundError,
    AuthorizationError,
    PyGridError,
    MissingRequestKeyError,
)
from ..database import Role, User
from .. import db


def create_role(current_user, private_key, role_fields):
    user_role = db.session.query(Role).get(current_user.role)

    if user_role is None:
        raise RoleNotFoundError
    if not user_role.can_edit_roles:
        raise AuthorizationError

    new_role = Role(**role_fields)
    db.session.add(new_role)
    db.session.commit()

    return new_role


def get_role(current_user, private_key, role_id):
    user_role = db.session.query(Role).get(current_user.role)

    if user_role is None:
        raise RoleNotFoundError
    if not user_role.can_edit_settings:
        raise AuthorizationError

    role = db.session.query(Role).get(role_id)
    if role is None:
        raise RoleNotFoundError

    return role


def get_all_roles(current_user, private_key):
    user_role = db.session.query(Role).get(current_user.role)

    if user_role is None:
        raise RoleNotFoundError
    if not user_role.can_edit_settings:
        raise AuthorizationError

    roles = db.session.query(Role).all()
    return roles


def put_role(current_user, role_id, new_fields):
    user_role = db.session.query(Role).get(current_user.role)

    if user_role is None:
        raise RoleNotFoundError
    if not user_role.can_edit_roles:
        raise AuthorizationError

    role = db.session.query(Role).get(role_id)
    if role is None:
        raise RoleNotFoundError

    for key, value in new_fields.items():
        setattr(role, key, value)

    db.session.commit()
    return role


def delete_role(current_user, role_id):
    user_role = db.session.query(Role).get(current_user.role)

    if user_role is None:
        raise RoleNotFoundError
    if not user_role.can_edit_roles:
        raise AuthorizationError

    role = db.session.query(Role).get(role_id)
    if role is None:
        raise RoleNotFoundError
    db.session.delete(role)
    db.session.commit()

    return role
