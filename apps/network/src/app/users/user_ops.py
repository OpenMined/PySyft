import logging
from json import dumps, loads
from secrets import token_hex

import jwt
from flask import current_app as app
from bcrypt import checkpw, gensalt, hashpw

from .. import db
from ..exceptions import (
    AuthorizationError,
    InvalidCredentialsError,
    MissingRequestKeyError,
    PyGridError,
    RoleNotFoundError,
    UserNotFoundError,
)
from ..database import Role, User


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

    user = db.session.query(User).filter_by(private_key=private_key).one_or_none()
    if user is None:
        raise UserNotFoundError

    user_role = db.session.query(Role).get(user.role)
    if user_role is None:
        raise RoleNotFoundError

    return user, user_role


def signup_user(private_key, email, password, role):
    user_role = None
    user = None

    try:
        user, user_role = identify_user(private_key)
    except Exception as e:
        logging.warning("Existing user could not be linked")

    if private_key is not None and (user is None or user_role is None):
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
    elif role is not None and user_role is not None and user_role.can_create_users:
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

    user = User.query.filter_by(email=email, private_key=private_key).first()
    if user is None:
        raise InvalidCredentialsError

    hashed = user.hashed_password.encode("UTF-8")
    salt = user.salt.encode("UTF-8")

    if checkpw(password, salt + hashed):
        token = jwt.encode({"id": user.id}, app.config["SECRET_KEY"])
        token = token.decode("UTF-8")
        return token
    else:
        raise InvalidCredentialsError


def get_all_users(current_user, private_key):
    user_role = Role.query.get(current_user.role)
    if user_role is None:
        raise RoleNotFoundError

    users = User.query.all()
    return users


def get_specific_user(current_user, private_key, user_id):
    user_role = Role.query.get(current_user.role)
    if user_role is None:
        raise RoleNotFoundError

    user = User.query.get(user_id)
    if user is None:
        raise UserNotFoundError

    return user


def change_user_email(current_user, private_key, email, user_id):
    user_role = db.session.query(Role).get(current_user.role)
    edited_user = db.session.query(User).get(user_id)

    if user_role is None:
        raise RoleNotFoundError
    if user_id != current_user.id and not user_role.can_create_users:
        raise AuthorizationError
    if edited_user is None:
        raise UserNotFoundError

    setattr(edited_user, "email", email)
    db.session.commit()

    return edited_user


def change_user_role(current_user, private_key, role, user_id):
    if user_id == 1:  # can't change Owner
        raise AuthorizationError

    user_role = db.session.query(Role).get(current_user.role)
    owner_role = db.session.query(User).get(1).id
    edited_user = db.session.query(User).get(user_id)

    if user_role is None:
        raise RoleNotFoundError
    if user_id != current_user.id and not user_role.can_create_users:
        raise AuthorizationError
    # Only Owners can create other Owners
    if role == owner_role and current_user.id != owner_role:
        raise AuthorizationError
    if edited_user is None:
        raise UserNotFoundError

    setattr(edited_user, "role", int(role))
    db.session.commit()

    return edited_user


def change_user_password(current_user, private_key, password, user_id):
    user_role = db.session.query(Role).get(current_user.role)
    edited_user = db.session.query(User).get(user_id)

    if user_role is None:
        raise RoleNotFoundError
    if user_id != current_user.id and not user_role.can_create_users:
        raise AuthorizationError
    if edited_user is None:
        raise UserNotFoundError

    salt, hashed = salt_and_hash_password(password, 12)
    setattr(edited_user, "salt", salt)
    setattr(edited_user, "hashed_password", hashed)
    db.session.commit()

    return edited_user


def delete_user(current_user, private_key, user_id):
    user_role = db.session.query(Role).get(current_user.role)
    edited_user = db.session.query(User).get(user_id)

    if user_role is None:
        raise RoleNotFoundError
    if user_id != current_user.id and not user_role.can_create_users:
        raise AuthorizationError
    if edited_user is None:
        raise UserNotFoundError

    db.session.delete(edited_user)
    db.session.commit()

    return edited_user


def search_users(current_user, private_key, filters):
    user_role = db.session.query(Role).get(current_user.role)

    if user_role is None:
        raise RoleNotFoundError

    query = db.session().query(User)
    for attr, value in filters.items():
        query = query.filter(getattr(User, attr).like("%%%s%%" % value))

    users = query.all()

    return users
