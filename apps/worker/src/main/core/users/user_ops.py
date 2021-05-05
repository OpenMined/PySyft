# stdlib
from datetime import datetime
from datetime import timedelta
from json import dumps
from json import loads
from json.decoder import JSONDecodeError
import logging
from secrets import token_hex

# third party
from bcrypt import checkpw
from bcrypt import gensalt
from bcrypt import hashpw
from flask import Response
from flask import current_app as app
from flask import request
import jwt
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# grid relative
from ..codes import RESPONSE_MSG
from ..database import Group
from ..database import Role
from ..database import User
from ..database import UserGroup
from ..database import db
from ..database import expand_user_object
from ..exceptions import AuthorizationError
from ..exceptions import GroupNotFoundError
from ..exceptions import InvalidCredentialsError
from ..exceptions import MissingRequestKeyError
from ..exceptions import PyGridError
from ..exceptions import RoleNotFoundError
from ..exceptions import UserNotFoundError
from ..node import node


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


def signup_user(email, password, role=None, private_key=None):
    user_role = None
    user = None

    try:
        user, user_role = identify_user(private_key)
    except Exception as e:
        logging.warning("Existing user could not be linked")

    if private_key is not None and (user is None or user_role is None):
        raise InvalidCredentialsError

    # generate a signing key
    private_key = SigningKey.generate()
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
            private_key=node.signing_key.encode(encoder=HexEncoder).decode("utf-8"),
            role=role,
        )
    elif role is not None and user_role is not None and user_role.can_create_users:
        if db.session.query(Role).get(role) is None:
            raise RoleNotFoundError
        new_user = User(
            email=email,
            hashed_password=hashed,
            salt=salt,
            private_key=private_key.encode(encoder=HexEncoder).decode("utf-8"),
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
            private_key=private_key.encode(encoder=HexEncoder).decode("utf-8"),
            role=role,
        )

    db.session.add(new_user)
    db.session.commit()

    # Add the new key in verify_key_registry
    node.guest_verify_key_registry.add(private_key.verify_key)
    user = expand_user_object(new_user)
    return {"user": user}


def login_user(email, password):
    password = password.encode("UTF-8")

    user = User.query.filter_by(email=email).first()

    if user is None:
        raise InvalidCredentialsError

    hashed = user.hashed_password.encode("UTF-8")
    salt = user.salt.encode("UTF-8")

    if checkpw(password, salt + hashed):
        token = jwt.encode({"id": user.id}, app.config["SECRET_KEY"])
        token = token.decode("UTF-8")
        return {
            "token": token,
            "key": user.private_key,
            "metadata": node.get_metadata_for_client()
            .serialize()
            .SerializeToString()
            .decode("ISO-8859-1"),
        }
    else:
        raise InvalidCredentialsError


def get_all_users(current_user):
    user_role = Role.query.get(current_user.role)
    if user_role is None:
        raise RoleNotFoundError

    if not user_role.can_triage_requests:
        raise AuthorizationError

    users = User.query.all()
    users = [expand_user_object(user) for user in users]
    return {"users": users}


def get_specific_user(current_user, user_id):
    user_id = int(user_id)
    user_role = Role.query.get(current_user.role)
    if user_role is None:
        raise RoleNotFoundError

    if not user_role.can_triage_requests:
        raise AuthorizationError

    user = User.query.get(user_id)
    if user is None:
        raise UserNotFoundError

    return {"user": expand_user_object(user)}


def change_user_email(current_user, email, user_id):
    user_id = int(user_id)
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

    return {"user": expand_user_object(edited_user)}


def change_user_role(current_user, role, user_id):
    if int(user_id) == 1:  # can't change Owner
        raise AuthorizationError

    user_role = db.session.query(Role).get(current_user.role)
    owner_role = db.session.query(User).get(1).id
    edited_user = db.session.query(User).get(user_id)

    if user_role is None:
        raise RoleNotFoundError
    if int(user_id) != current_user.id and not user_role.can_create_users:
        raise AuthorizationError
    # Only Owners can create other Owners
    if role == owner_role and current_user.id != owner_role:
        raise AuthorizationError
    if edited_user is None:
        raise UserNotFoundError

    setattr(edited_user, "role", int(role))
    db.session.commit()

    return {"user": expand_user_object(edited_user)}


def change_user_password(current_user, password, user_id):
    user_id = int(user_id)
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

    return {"user": expand_user_object(edited_user)}


def change_user_groups(current_user, groups, user_id):
    user_id = int(user_id)
    user_role = db.session.query(Role).get(current_user.role)
    edited_user = db.session.query(User).get(user_id)

    if user_role is None:
        raise RoleNotFoundError
    if user_id != current_user.id and not user_role.can_create_users:
        raise AuthorizationError
    if edited_user is None:
        raise UserNotFoundError

    query = db.session().query
    user_groups = query(UserGroup).filter_by(user=user_id).all()

    for group in user_groups:
        db.session.delete(group)

    for new_group in groups:
        if query(Group.id).filter_by(id=new_group).scalar() is None:
            raise GroupNotFoundError
        new_usergroup = UserGroup(user=user_id, group=new_group)
        db.session.add(new_usergroup)

    db.session.commit()

    return {"user": expand_user_object(edited_user)}


def delete_user(current_user, user_id):
    user_id = int(user_id)
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

    return {"user": expand_user_object(edited_user)}


def search_users(current_user, filters):
    user_role = db.session.query(Role).get(current_user.role)

    if user_role is None:
        raise RoleNotFoundError
    if not user_role.can_triage_requests:
        raise AuthorizationError

    query = db.session().query(User)
    for attr, value in filters.items():
        if attr != "group":
            query = query.filter(getattr(User, attr).like("%%%s%%" % value))
        else:
            query = query.join(UserGroup).filter(UserGroup.group.in_([group]))

    users = query.all()
    users = [expand_user_object(user) for user in users]
    return {"users": users}
