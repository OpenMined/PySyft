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
owner_role = ("Owner", True, True, True, True)
user_role = ("User", False, False, False, False)
admin_role = ("Administrator", True, True, False, False)

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


def test_initial_setup(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    database.session.commit()

    result = client.post(
        "/setup/",
        json={
            "email": "ionesio@email.com",
            "password": "testing",
            "node_name": "OpenMined Node",
        },
    )
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Running initial setup!"}


def test_get_setup(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    client.post(
        "/setup/",
        json={
            "email": "ionesio@email.com",
            "password": "testing",
            "node_name": "OpenMined Node",
        },
    )

    result = client.get(
        "/setup/",
        headers=headers,
    )

    assert result.status_code == 200
    assert result.get_json() == {
        "id": 1,
        "node_name": "OpenMined Node",
        "private_key": "",
        "aws_credentials": "",
        "gcp_credentials": "",
        "azure_credentials": "",
        "cache_strategy": "",
        "replicate_db": False,
        "auto_scale": "",
        "tensor_expiration_policy": 0,
        "allow_user_signup": False,
    }
