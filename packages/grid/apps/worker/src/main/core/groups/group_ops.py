# stdlib
from datetime import datetime
from datetime import timedelta
from json import dumps
from json import loads
from json.decoder import JSONDecodeError
import logging
from secrets import token_hex

from main.core.codes import RESPONSE_MSG
from main.core.database import Group
from main.core.database import Role
from main.core.database import User
from main.core.database import UserGroup
from main.core.database import db
from main.core.database.utils import model_to_json
from main.core.exceptions import AuthorizationError
from main.core.exceptions import GroupNotFoundError
from main.core.exceptions import InvalidCredentialsError
from main.core.exceptions import MissingRequestKeyError
from main.core.exceptions import PyGridError
from main.core.exceptions import RoleNotFoundError
from main.core.exceptions import UserNotFoundError


def create_group(current_user, name):
    user_role = Role.query.get(current_user.role)
    if user_role is None:
        raise RoleNotFoundError

    if not user_role.can_create_groups:
        raise AuthorizationError

    new_group = Group(name=name)
    db.session.add(new_group)
    db.session.commit()

    return model_to_json(new_group)


def get_group(current_user, group_id):
    user_role = Role.query.get(current_user.role)
    if user_role is None:
        raise RoleNotFoundError

    if not user_role.can_triage_requests:
        raise AuthorizationError

    group = Group.query.get(group_id)
    if group is None:
        raise GroupNotFoundError

    return model_to_json(group)


def get_all_groups(current_user):
    user_role = Role.query.get(current_user.role)
    if user_role is None:
        raise RoleNotFoundError

    if not user_role.can_triage_requests:
        raise AuthorizationError

    groups = Group.query.all()
    groups = [model_to_json(g) for g in groups]
    return groups


def put_group(current_user, group_id, new_fields):
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
    return model_to_json(group)


def delete_group(current_user, group_id):
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

    return model_to_json(group)
