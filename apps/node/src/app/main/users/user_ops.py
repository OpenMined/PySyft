import logging
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


def salt_and_hash_password(password, rounds):
    password = password.encode("UTF-8")
    salt = gensalt(rounds=rounds)
    hashed = hashpw(password, salt)
    hashed = hashed[len(salt) :]
    hashed = hashed.decode("UTF-8")
    salt = salt.decode("UTF-8")
    return salt, hashed


def identify_user(private_key):
    if private_key is None:
        raise MissingRequestKeyError

    usr = db.session.query(User).filter_by(private_key=private_key).one_or_none()
    if usr is None:
        raise UserNotFoundError

    usr_role = db.session.query(Role).get(usr.role)
    if usr_role is None:
        raise RoleNotFoundError

    return usr, usr_role


def signup_user(private_key, email, password, role):
    usr_role = None
    usr = None

    try:
        usr, usr_role = identify_user(private_key)
    except Exception as e:
        logging.warning("Existing user could not be linked")

    if private_key is not None and (usr is None or usr_role is None):
        raise InvalidCredentialsError

    private_key = token_hex(32)
    salt, hashed = salt_and_hash_password(password, 12)
    no_user = len(db.session.query(User).all()) == 0

    if no_user:
        role = db.session.query(Role.id).filter_by(name="Owner").first()
        if role is None:
            raise RoleNotFoundError
        role = role[0]
        new_user = User(
            email=email,
            hashed_password=hashed,
            salt=salt,
            private_key=private_key,
            role=role,
        )
    elif role is not None and usr_role is not None and usr_role.can_create_users:
        if db.session.query(Role).get(role) is None:
            raise RoleNotFoundError
        new_user = User(
            email=email,
            hashed_password=hashed,
            salt=salt,
            private_key=private_key,
            role=role,
        )
    else:
        role = db.session.query(Role.id).filter_by(name="User").first()
        if role is None:
            raise RoleNotFoundError
        role = role[0]
        new_user = User(
            email=email,
            hashed_password=hashed,
            salt=salt,
            private_key=private_key,
            role=role,
        )

    db.session.add(new_user)
    db.session.commit()

    return new_user


def login_user(private_key, email, password):
    password = password.encode("UTF-8")

    usr = User.query.filter_by(email=email, private_key=private_key).first()
    if usr is None:
        raise InvalidCredentialsError

    hashed = usr.hashed_password.encode("UTF-8")
    salt = usr.salt.encode("UTF-8")

    if checkpw(password, salt + hashed):
        token = jwt.encode({"id": usr.id}, app.config["SECRET_KEY"])
        token = token.decode("UTF-8")
        return token
    else:
        raise InvalidCredentialsError


def get_all_users(current_user, private_key):
    usr_role = Role.query.get(current_user.role)
    if usr_role is None:
        raise RoleNotFoundError

    if not usr_role.can_triage_jobs:
        raise AuthorizationError

    users = User.query.all()
    return users


def get_specific_user(current_user, private_key, user_id):
    usr_role = Role.query.get(current_user.role)
    if usr_role is None:
        raise RoleNotFoundError

    if not usr_role.can_triage_jobs:
        raise AuthorizationError

    user = User.query.get(user_id)
    if user is None:
        raise UserNotFoundError

    return user


def change_usr_email(current_user, private_key, email, user_id):
    usr_role = db.session.query(Role).get(current_user.role)
    edited_user = db.session.query(User).get(user_id)

    if usr_role is None:
        raise RoleNotFoundError
    if user_id != current_user.id and not usr_role.can_create_users:
        raise AuthorizationError
    if edited_user is None:
        raise UserNotFoundError

    setattr(edited_user, "email", email)
    db.session.commit()

    return edited_user


def change_usr_role(current_user, private_key, role, user_id):
    if user_id == 1:  # can't change Owner
        raise AuthorizationError

    usr_role = db.session.query(Role).get(current_user.role)
    owner_role = db.session.query(User).get(1).id
    edited_user = db.session.query(User).get(user_id)

    if usr_role is None:
        raise RoleNotFoundError
    if user_id != current_user.id and not usr_role.can_create_users:
        raise AuthorizationError
    # Only Owners can create other Owners
    if role == owner_role and current_user.id != owner_role:
        raise AuthorizationError
    if edited_user is None:
        raise UserNotFoundError

    setattr(edited_user, "role", int(role))
    db.session.commit()

    return edited_user


def change_usr_password(current_user, private_key, password, user_id):
    usr_role = db.session.query(Role).get(current_user.role)
    edited_user = db.session.query(User).get(user_id)

    if usr_role is None:
        raise RoleNotFoundError
    if user_id != current_user.id and not usr_role.can_create_users:
        raise AuthorizationError
    if edited_user is None:
        raise UserNotFoundError

    salt, hashed = salt_and_hash_password(password, 12)
    setattr(edited_user, "salt", salt)
    setattr(edited_user, "hashed_password", hashed)
    db.session.commit()

    return edited_user


def change_usr_groups(current_user, private_key, groups, user_id):
    usr_role = db.session.query(Role).get(current_user.role)
    edited_user = db.session.query(User).get(user_id)

    if usr_role is None:
        raise RoleNotFoundError
    if user_id != current_user.id and not usr_role.can_create_users:
        raise AuthorizationError
    if edited_user is None:
        raise UserNotFoundError

    query = db.session().query
    usr_groups = query(UserGroup).filter_by(user=user_id).all()

    for group in usr_groups:
        db.session.delete(group)

    for new_group in groups:
        if query(Group.id).filter_by(id=new_group).scalar() is None:
            raise GroupNotFoundError
        new_usrgroup = UserGroup(user=user_id, group=new_group)
        db.session.add(new_usrgroup)

    db.session.commit()

    return edited_user


def delete_user(current_user, private_key, user_id):
    usr_role = db.session.query(Role).get(current_user.role)
    edited_user = db.session.query(User).get(user_id)

    if usr_role is None:
        raise RoleNotFoundError
    if user_id != current_user.id and not usr_role.can_create_users:
        raise AuthorizationError
    if edited_user is None:
        raise UserNotFoundError

    db.session.delete(edited_user)
    db.session.commit()

    return edited_user


def search_users(current_user, private_key, filters, group):
    usr_role = db.session.query(Role).get(current_user.role)

    if usr_role is None:
        raise RoleNotFoundError
    if not usr_role.can_triage_jobs:
        raise AuthorizationError

    query = db.session().query(User)
    for attr, value in filters.items():
        if attr != "group":
            query = query.filter(getattr(User, attr).like("%%%s%%" % value))
        else:
            query = query.join(UserGroup).filter(UserGroup.group.in_([group]))

    users = query.all()

    return users
