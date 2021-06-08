# stdlib
from json import dumps
from json import loads

# third party
from flask import current_app as app
import jwt
import pytest
from src.main.core.database import *

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


@pytest.mark.skip(reason="changes in association requets still in progress")
def test_send_association_request(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    result = client.post(
        "/association-requests/request",
        data={"id": "54623156", "address": "159.15.223.162"},
        headers=headers,
    )
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Association request sent!"}


@pytest.mark.skip(reason="changes in association requets still in progress")
def test_receive_association_request(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.post(
        "/association-requests/receive",
        data={"id": "54623156", "address": "159.15.223.162"},
        headers=headers,
    )

    assert result.status_code == 200
    assert result.get_json() == {"msg": "Association request received!"}


@pytest.mark.skip(reason="changes in association requets still in progress")
def test_reply_association_request(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.post(
        "/association-requests/respond",
        data={"id": "54623156", "address": "159.15.223.162"},
        headers=headers,
    )
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Association request was replied!"}


@pytest.mark.skip(reason="changes in association requets still in progress")
def test_get_all_association_requests(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get("/association-requests/", headers=headers)
    assert result.status_code == 200
    assert result.get_json() == {
        "association-requests": ["Network A", "Network B", "Network C"]
    }


@pytest.mark.skip(reason="changes in association requets still in progress")
def test_get_specific_association_requests(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get("/association-requests/51613546", headers=headers)
    assert result.status_code == 200
    assert result.get_json() == {
        "association-request": {
            "ID": "51613546",
            "address": "156.89.33.200",
        }
    }


@pytest.mark.skip(reason="changes in association requets still in progress")
def test_delete_association_requests(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/association-requests/51661659", headers=headers)
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Association request deleted!"}
