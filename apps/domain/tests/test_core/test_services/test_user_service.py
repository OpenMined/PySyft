from src.main.core.database import *
from src.main.core.manager import UserManager
from src.main.core.exceptions import InvalidCredentialsError
from src.main.core.node import GridDomain
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import pytest
from bcrypt import checkpw

from syft.grid.messages.user_messages import CreateUserMessage

owner = ("Owner", True, True, True, True, True, True, True)
user = ("User", False, False, False, False, False, False, False)
admin = ("Administrator", True, True, True, True, False, False, True)

# generate a signing key
generic_key = SigningKey.generate()


@pytest.fixture
def cleanup(database):
    yield
    try:
        database.session.query(User).delete()
        database.session.query(Role).delete()
        database.session.query(Group).delete()
        database.session.query(UserGroup).delete()
        database.session.commit()
    except:
        database.session.rollback()


def __create_roles(database):
    owner_role = create_role(*owner)
    user_role = create_role(*user)
    admin_role = create_role(*admin)
    database.session.add(owner_role)
    database.session.add(user_role)
    database.session.add(admin_role)
    database.session.commit()


def build_signup_syft_msg(node, msg, key):
    content = {
        "address": node.address,
        "content": msg,
        "reply_to": node.address,
    }
    signed_msg = CreateUserMessage(**content).sign(signing_key=key)
    return node.recv_immediate_msg_with_reply(msg=signed_msg).message


def test_create_user_without_email(database, domain, cleanup):
    __create_roles(database)

    request_content = {
        "password": "owner123",
    }

    response = build_signup_syft_msg(domain, request_content, generic_key)

    # Check message response
    assert response.success == False
    assert response.content == {
        "error": "Invalid request payload, empty fields (email/password)!"
    }


def test_create_user_without_password(database, domain, cleanup):
    __create_roles(database)

    request_content = {
        "email": "owner@gmail.com",
    }

    response = build_signup_syft_msg(domain, request_content, generic_key)

    # Check message response
    assert response.success == False
    assert response.content == {
        "error": "Invalid request payload, empty fields (email/password)!"
    }


def test_create_first_user_msg(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)

    request_content = {
        "email": "owner@gmail.com",
        "password": "owner123",
    }

    response = build_signup_syft_msg(domain, request_content, generic_key)

    # Check message response
    assert response.success == True
    assert response.content == {"msg": "User created successfully!"}

    # Check database
    assert len(users) == 1
    user = users.query(email="owner@gmail.com")[0]

    assert user.email == "owner@gmail.com"
    assert users.login(email="owner@gmail.com", password="owner123")
    assert users.role(user_id=user.id).name == "Owner"


def test_create_second_user(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)

    first_user_content = {
        "email": "owner@gmail.com",
        "password": "owner123",
    }

    response = build_signup_syft_msg(domain, first_user_content, generic_key)

    # Check message response
    assert response.success == True
    assert response.content == {"msg": "User created successfully!"}

    # Check database
    assert len(users) == 1
    user = users.query(email="owner@gmail.com")[0]

    assert user.email == "owner@gmail.com"
    assert users.login(email="owner@gmail.com", password="owner123")
    assert users.role(user_id=user.id).name == "Owner"
    assert users.role(user_id=user.id).can_create_users
    assert users.role(user_id=user.id).can_triage_requests

    second_user_content = {
        "email": "stduser@gmail.com",
        "password": "stduser123",
    }

    response = build_signup_syft_msg(domain, second_user_content, generic_key)

    # Check message response
    assert response.success == True
    assert response.content == {"msg": "User created successfully!"}

    # Check database
    assert len(users) == 2
    user = users.query(email="stduser@gmail.com")[0]

    assert user.email == "stduser@gmail.com"
    assert users.login(email="stduser@gmail.com", password="stduser123")
    assert users.role(user_id=user.id).name == "User"
    assert not users.role(user_id=user.id).can_create_users
    assert not users.role(user_id=user.id).can_triage_requests


def test_create_second_user_with_invalid_role_name(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)

    owner_content = {
        "email": "owner@gmail.com",
        "password": "owner123",
    }

    response = build_signup_syft_msg(domain, owner_content, generic_key)

    # Check message response
    assert response.success == True
    assert response.content == {"msg": "User created successfully!"}

    # Check database
    assert len(users) == 1
    user = users.query(email="owner@gmail.com")[0]

    assert user.email == "owner@gmail.com"
    assert users.login(email="owner@gmail.com", password="owner123")
    assert users.role(user_id=user.id).name == "Owner"
    assert users.role(user_id=user.id).can_create_users
    assert users.role(user_id=user.id).can_triage_requests

    owner_id = str(users.query(email="owner@gmail.com")[0].id)

    second_user_content = {
        "email": "admin_user@gmail.com",
        "password": "admin_user123",
        "role": "Random Role",
        "current_user": owner_id,
    }

    response = build_signup_syft_msg(domain, second_user_content, generic_key)

    # Check message response
    assert response.success == False
    assert response.content == {"error": "Role not found!"}

    # Check database
    assert len(users) == 1


def test_create_second_user_with_owner_role_name(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)

    owner_content = {
        "email": "owner@gmail.com",
        "password": "owner123",
    }

    response = build_signup_syft_msg(domain, owner_content, generic_key)

    # Check message response
    assert response.success == True
    assert response.content == {"msg": "User created successfully!"}

    # Check database
    assert len(users) == 1
    user = users.query(email="owner@gmail.com")[0]

    assert user.email == "owner@gmail.com"
    assert users.login(email="owner@gmail.com", password="owner123")
    assert users.role(user_id=user.id).name == "Owner"
    assert users.role(user_id=user.id).can_create_users
    assert users.role(user_id=user.id).can_triage_requests

    owner_id = str(users.query(email="owner@gmail.com")[0].id)

    second_user_content = {
        "email": "admin_user@gmail.com",
        "password": "admin_user123",
        "role": "Owner",
        "current_user": owner_id,
    }

    response = build_signup_syft_msg(domain, second_user_content, generic_key)

    # Check message response
    assert response.success == False
    assert response.content == {
        "error": 'You can\'t create a new User with "Owner" role!'
    }

    # Check database
    assert len(users) == 1


def test_create_second_user_with_role_and_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)

    owner_content = {
        "email": "owner@gmail.com",
        "password": "owner123",
    }

    response = build_signup_syft_msg(domain, owner_content, generic_key)

    # Check message response
    assert response.success == True
    assert response.content == {"msg": "User created successfully!"}

    # Check database
    assert len(users) == 1
    user = users.query(email="owner@gmail.com")[0]

    assert user.email == "owner@gmail.com"
    assert users.login(email="owner@gmail.com", password="owner123")
    assert users.role(user_id=user.id).name == "Owner"
    assert users.role(user_id=user.id).can_create_users
    assert users.role(user_id=user.id).can_triage_requests

    owner_id = str(users.query(email="owner@gmail.com")[0].id)

    second_user_content = {
        "email": "admin_user@gmail.com",
        "password": "admin_user123",
        "role": "Administrator",
        "current_user": owner_id,
    }

    response = build_signup_syft_msg(domain, second_user_content, generic_key)

    # Check message response
    assert response.success == True
    assert response.content == {"msg": "User created successfully!"}

    # Check database
    assert len(users) == 2
    user = users.query(email="admin_user@gmail.com")[0]

    assert user.email == "admin_user@gmail.com"
    assert users.login(email="admin_user@gmail.com", password="admin_user123")
    assert users.role(user_id=user.id).name == "Administrator"
    assert users.role(user_id=user.id).can_create_users == True
    assert users.role(user_id=user.id).can_triage_requests == True


def test_create_third_user_without_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)

    first_user_content = {
        "email": "owner@gmail.com",
        "password": "owner123",
    }

    response = build_signup_syft_msg(domain, first_user_content, generic_key)

    # Check message response
    assert response.success == True
    assert response.content == {"msg": "User created successfully!"}

    # Check database
    assert len(users) == 1
    user = users.query(email="owner@gmail.com")[0]

    assert user.email == "owner@gmail.com"
    assert users.login(email="owner@gmail.com", password="owner123")
    assert users.role(user_id=user.id).name == "Owner"
    assert users.role(user_id=user.id).can_create_users
    assert users.role(user_id=user.id).can_triage_requests

    second_user_content = {
        "email": "stduser@gmail.com",
        "password": "stduser123",
    }

    response = build_signup_syft_msg(domain, second_user_content, generic_key)

    # Check message response
    assert response.success == True
    assert response.content == {"msg": "User created successfully!"}

    # Check database
    assert len(users) == 2
    user = users.query(email="stduser@gmail.com")[0]

    assert user.email == "stduser@gmail.com"
    assert users.login(email="stduser@gmail.com", password="stduser123")
    assert users.role(user_id=user.id).name == "User"
    assert not users.role(user_id=user.id).can_create_users
    assert not users.role(user_id=user.id).can_triage_requests

    second_user_content = {
        "email": "stduser2@gmail.com",
        "password": "stduser2123",
        "role": "Administrator",
        "current_user": str(user.id),
    }

    response = build_signup_syft_msg(domain, second_user_content, generic_key)

    # Check message response
    assert response.success == True
    assert response.content == {"msg": "User created successfully!"}

    # Check database
    assert len(users) == 3
    user = users.query(email="stduser2@gmail.com")[0]

    assert user.email == "stduser2@gmail.com"
    assert users.login(email="stduser2@gmail.com", password="stduser2123")
    assert users.role(user_id=user.id).name == "User"
    assert not users.role(user_id=user.id).can_create_users
    assert not users.role(user_id=user.id).can_triage_requests
