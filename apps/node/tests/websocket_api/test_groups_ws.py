from json import dumps, loads

import jwt
import pytest
from flask import current_app as app
from src.app.main.database import *
from src.app.main.events.group_related import *

JSON_DECODE_ERR_MSG = (
    "Expecting property name enclosed in " "double quotes: line 1 column 2 (char 1)"
)
owner_role = ("Owner", True, True, True, True, True, True, True)
user_role = ("User", False, False, False, False, False, False, False)
admin_role = ("Administrator", True, True, True, True, False, False, True)

user1 = (
    "tech@gibberish.com",
    "BDEB6E8EE39B6C70835993486C9E65DC",
    "]GBF[R>GX[9Cmk@DthFT!mhloUc%[f",
    "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
    1,
)
user2 = (
    "anemail@anemail.com",
    "2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
    "$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
    "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
    2,
)
user3 = (
    "anemail@anemail.com",
    "2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
    "$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
    "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
    3,
)
user4 = (
    "tech@gibberish.com",
    "2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
    "$2b$12$tufn64/0gSIAdprqBrRzC.",
    "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
    2,
)


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


# CREATE GROUP


def test_create_group_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "name": "Hospital X",
    }
    result = create_group_socket(message)
    result = loads(result)

    assert result["group"]["id"] == 1
    assert result["group"]["name"] == "Hospital X"


def test_create_group_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {"token": token.decode("UTF-8"), "name": "Hospital X"}
    result = create_group_socket(message)
    result = loads(result)

    assert result["error"] == "Missing request key!"


def test_create_group_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "name": "Hospital X",
    }
    result = create_group_socket(message)
    result = loads(result)

    assert result["error"] == "Missing request key!"


def test_create_group_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "invalid312987as12they0come",
        "token": token.decode("UTF-8"),
        "name": "Hospital X",
    }
    result = create_group_socket(message)
    result = loads(result)

    assert result["error"] == "Invalid credentials!"


def test_create_group_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "peppperplsiwouldhavesome")
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "name": "Hospital X",
    }
    result = create_group_socket(message)
    result = loads(result)

    assert result["error"] == "Invalid credentials!"


def test_create_group_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "name": "Hospital X",
    }
    result = create_group_socket(message)
    result = loads(result)

    assert result["error"] == "User is not authorized for this operation!"


# GET GROUP


def test_get_group_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
    }
    result = get_group_socket(message)
    result = loads(result)

    assert result["group"]["id"] == 1
    assert result["group"]["name"] == "Hospital X"


def test_get_group_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {"token": token.decode("UTF-8"), "id": 1}
    result = get_group_socket(message)
    result = loads(result)

    assert result["error"] == "Missing request key!"


def test_get_group_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "id": 1,
    }
    result = get_group_socket(message)
    result = loads(result)

    assert result["error"] == "Missing request key!"


def test_get_group_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "invalid85b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
    }
    result = get_group_socket(message)
    result = loads(result)

    assert result["error"] == "Invalid credentials!"


def test_get_group_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, "thewrongsecret")
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
    }
    result = get_group_socket(message)
    result = loads(result)

    assert result["error"] == "Invalid credentials!"


def test_get_group_unauthorized(client, database, cleanup):
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
    }
    result = get_group_socket(message)
    result = loads(result)

    assert result["error"] == "User is not authorized for this operation!"


def test_get_group_missing_group(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
    }
    result = get_group_socket(message)
    result = loads(result)

    assert result["error"] == "Group ID not found!"


# GET ALL GROUPS


def test_get_all_groups_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)
    new_group = Group(name="Hospital Y")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
    }
    result = get_all_groups_socket(message)
    result = loads(result)

    assert result["groups"][0]["id"] == 1
    assert result["groups"][0]["name"] == "Hospital X"
    assert result["groups"][1]["id"] == 2
    assert result["groups"][1]["name"] == "Hospital Y"


def test_get_all_groups_empty_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
    }
    result = get_all_groups_socket(message)
    result = loads(result)

    assert result["groups"] == []


def test_get_all_groups_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)
    new_group = Group(name="Hospital Y")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "token": token.decode("UTF-8"),
    }
    result = get_all_groups_socket(message)
    result = loads(result)

    assert result["error"] == "Missing request key!"


def test_get_all_groups_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)
    new_group = Group(name="Hospital Y")
    database.session.add(new_group)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
    }
    result = get_all_groups_socket(message)
    result = loads(result)

    assert result["error"] == "Missing request key!"


def test_get_all_groups_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)
    new_group = Group(name="Hospital Y")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "invalid85b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
    }
    result = get_all_groups_socket(message)
    result = loads(result)

    assert result["error"] == "Invalid credentials!"


def test_get_all_groups_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)
    new_group = Group(name="Hospital Y")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, "thewrongsecret")
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
    }
    result = get_all_groups_socket(message)
    result = loads(result)

    assert result["error"] == "Invalid credentials!"


def test_get_all_groups_unauthorized(client, database, cleanup):
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)
    new_group = Group(name="Hospital Y")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
    }
    result = get_all_groups_socket(message)
    result = loads(result)

    assert result["error"] == "User is not authorized for this operation!"


# PUT GROUP


def test_put_group_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    new_name = "Brand New Hospital A"
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "group": {"name": new_name},
        "id": 1,
    }
    result = put_group_socket(message)
    result = loads(result)

    assert result["group"]["id"] == 1
    assert result["group"]["name"] == new_name
    assert database.session.query(Group).get(1).name == new_name


def test_put_group_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    new_name = "Brand New Hospital A"
    message = {"token": token.decode("UTF-8"), "group": {"name": new_name}, "id": 1}
    result = put_group_socket(message)
    result = loads(result)

    assert result["error"] == "Missing request key!"


def test_put_group_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    new_name = "Brand New Hospital A"
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "group": {"name": new_name},
        "id": 1,
    }
    result = put_group_socket(message)
    result = loads(result)

    assert result["error"] == "Missing request key!"


def test_put_group_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    new_name = "Brand New Hospital A"
    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "invalid85b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
        "group": {"name": new_name},
        "id": 1,
    }

    result = put_group_socket(message)
    result = loads(result)

    assert result["error"] == "Invalid credentials!"


def test_put_group_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, "thewrongsecret")
    new_name = "Brand New Hospital A"
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
        "group": {"name": new_name},
    }
    result = put_group_socket(message)
    result = loads(result)

    assert result["error"] == "Invalid credentials!"


def test_put_group_unauthorized(client, database, cleanup):
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    new_name = "Brand New Hospital A"
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
        "group": {"name": new_name},
    }
    result = put_group_socket(message)
    result = loads(result)

    assert result["error"] == "User is not authorized for this operation!"


def test_put_group_missing_group(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    new_name = "Brand New Hospital A"
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
        "group": {"name": new_name},
    }
    result = put_group_socket(message)
    result = loads(result)

    assert result["error"] == "Group ID not found!"


# DELETE GROUP


def test_delete_group_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    assert database.session.query(Group).get(1) is not None

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "id": 1,
        "token": token.decode("UTF-8"),
    }
    result = delete_group_socket(message)
    result = loads(result)

    assert database.session.query(Group).get(1) is None


def test_delete_group_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {"token": token.decode("UTF-8"), "id": 1}
    result = delete_group_socket(message)
    result = loads(result)

    assert result["error"] == "Missing request key!"


def test_delete_group_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "id": 1,
    }
    result = delete_group_socket(message)
    result = loads(result)

    assert result["error"] == "Missing request key!"


def test_delete_group_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "invalid85b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
    }
    result = delete_group_socket(message)
    result = loads(result)

    assert result["error"] == "Invalid credentials!"


def test_put_delete_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, "thewrongsecret")
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
    }
    result = delete_group_socket(message)
    result = loads(result)

    assert result["error"] == "Invalid credentials!"


def test_delete_group_unauthorized(client, database, cleanup):
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_group = Group(name="Hospital X")
    database.session.add(new_group)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
    }
    result = delete_group_socket(message)
    result = loads(result)

    assert result["error"] == "User is not authorized for this operation!"


def test_delete_group_missing_group(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 1,
    }
    result = delete_group_socket(message)
    result = loads(result)

    assert result["error"] == "Group ID not found!"
