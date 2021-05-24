# stdlib
from json import dumps
from json import loads
import time

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
        database.session.query(BinObject).delete()
        database.session.query(ObjectMetadata).delete()
        database.session.commit()
    except:
        database.session.rollback()


def test_get_all_tensors(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    # Register the first tensor
    tags = ["#first-tensor"]
    description = "First tensor description"

    result = client.post(
        "/data-centric/tensors",
        json={
            "tensor": [1, 2, 3, 4, 5, 6],
            "description": description,
            "tags": tags,
            "searchable": True,
        },
        headers=headers,
    )
    tensor_id = result.get_json()["tensor_id"]

    # Register the second tensor
    tags = ["#second-tensor"]
    description = "Second tensor description"

    result = client.post(
        "/data-centric/tensors",
        json={
            "tensor": [1, 2, 3, 4, 5, 6],
            "description": description,
            "tags": tags,
            "searchable": True,
        },
        headers=headers,
    )
    tensor_id = result.get_json()["tensor_id"]

    # Test Get tensors request
    result = client.get("/data-centric/tensors", headers=headers)

    assert result.status_code == 200
    assert len(result.get_json()["tensors"]) == 2


def test_create_tensor(client, database, cleanup):
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
        "/data-centric/tensors",
        json={
            "tensor": [1, 2, 3, 4, 5, 6, 7, 8],
            "description": "A tensor sample",
            "tags": ["#x-tensor", "tensor-sample"],
            "searchable": True,
        },
        headers=headers,
    )

    assert result.status_code == 201
    assert result.get_json()["msg"] == "Tensor created succesfully!"


def test_get_specific_tensor(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    # Register a new tensor
    tags = ["#get-tensor"]
    description = "Get tensor test"

    result = client.post(
        "/data-centric/tensors",
        json={
            "tensor": [1, 2, 3, 4, 5, 6],
            "description": description,
            "tags": tags,
            "searchable": True,
        },
        headers=headers,
    )
    tensor_id = result.get_json()["tensor_id"]

    result = client.get("/data-centric/tensors/" + tensor_id, headers=headers)
    assert result.status_code == 200
    tensor = result.get_json()["tensor"]
    assert tensor["tags"] == tags
    assert tensor["description"] == description


def test_update_tensor(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    # Register a new tensor
    tags = ["#get-tensor"]
    description = "Get tensor test"

    result = client.post(
        "/data-centric/tensors",
        json={
            "tensor": [1, 2, 3, 4, 5, 6],
            "description": description,
            "tags": tags,
            "searchable": True,
        },
        headers=headers,
    )
    tensor_id = result.get_json()["tensor_id"]

    # Assert registered tensor metadata
    result = client.get("/data-centric/tensors/" + tensor_id, headers=headers)
    assert result.status_code == 200
    tensor = result.get_json()["tensor"]
    assert tensor["tags"] == tags
    assert tensor["description"] == description

    # Update tensor values
    modified_tags = ["#modified-tensor"]
    modified_description = "modified tensor test"

    # Update an existent tensor
    result = client.put(
        "/data-centric/tensors/" + tensor_id,
        json={
            "tensor": [1, 2, 5, 6],
            "description": modified_description,
            "tags": modified_tags,
            "searchable": True,
        },
        headers=headers,
    )
    assert result.status_code == 204

    # Assert updated tensor metadata
    result = client.get("/data-centric/tensors/" + tensor_id, headers=headers)
    assert result.status_code == 200
    tensor = result.get_json()["tensor"]
    assert tensor["tags"] == modified_tags
    assert tensor["description"] == modified_description


def test_delete_tensor(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    # Register a new tensor
    tags = ["#get-tensor"]
    description = "Get tensor test"

    result = client.post(
        "/data-centric/tensors",
        json={
            "tensor": [1, 2, 3, 4, 5, 6],
            "description": description,
            "tags": tags,
            "searchable": True,
        },
        headers=headers,
    )

    tensor_id = result.get_json()["tensor_id"]

    result = client.delete("/data-centric/tensors/" + tensor_id, headers=headers)

    assert result.status_code == 204
