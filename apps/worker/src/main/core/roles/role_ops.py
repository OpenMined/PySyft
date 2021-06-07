# stdlib
from json import dumps
from json import loads
from json.decoder import JSONDecodeError
import logging

# third party
from flask import Response
from flask import request
from nacl.signing import SigningKey

# grid relative
from ..codes import RESPONSE_MSG
from ..database import Role
from ..database import User
from ..database import db
from ..database.utils import model_to_json
from ..exceptions import AuthorizationError
from ..exceptions import MissingRequestKeyError
from ..exceptions import PyGridError
from ..exceptions import RoleNotFoundError
from ..exceptions import UserNotFoundError

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


def create_role(current_user, role_fields, private_key=None):
    user_role = db.session.query(Role).get(current_user.role)
    private_key = SigningKey.generate()

    for field in expected_fields:
        if role_fields.get(field) is None:
            raise MissingRequestKeyError

    if user_role is None:
        raise RoleNotFoundError
    if not user_role.can_edit_roles:
        raise AuthorizationError

    new_role = Role(**role_fields)
    db.session.add(new_role)
    db.session.commit()

    return model_to_json(new_role)


def get_role(current_user, role_id):
    user_role = db.session.query(Role).get(current_user.role)

    if user_role is None:
        raise RoleNotFoundError
    if not user_role.can_triage_requests:
        raise AuthorizationError

    role = db.session.query(Role).get(role_id)
    if role is None:
        raise RoleNotFoundError

    return model_to_json(role)


def get_all_roles(current_user):
    user_role = db.session.query(Role).get(current_user.role)

    if user_role is None:
        raise RoleNotFoundError
    if not user_role.can_triage_requests:
        raise AuthorizationError

    roles = db.session.query(Role).all()
    roles = [model_to_json(r) for r in roles]
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
    return model_to_json(role)


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

    return model_to_json(role)
