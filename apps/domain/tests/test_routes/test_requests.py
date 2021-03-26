from json import dumps, loads

import jwt
import pytest
from flask import current_app as app

from src.main.core.database import *
import time

owner_role = ("Owner", True, True, True, True, True, True, True)

user1 = (
    "tech@gibberish.com",
    "BDEB6E8EE39B6C70835993486C9E65DC",
    "]GBF[R>GX[9Cmk@DthFT!mhloUc%[f",
    "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
    1,
)


@pytest.fixture
def cleanup(database):
    yield
    try:
        database.session.query(User).delete()
        database.session.query(Role).delete()
        database.session.query(Group).delete()
        database.session.query(UserGroup).delete()
        database.session.query(Request).delete()
        database.session.commit()
    except:
        database.session.rollback()


@pytest.mark.skip(reason="This test need to be updated!")
def test_create_request(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    object_id = "61612325"
    reason = "sample reason"
    request_type = "permissions"

    result = client.post(
        "/dcfl/requests",
        json={
            "object_id": object_id,
            "reason": reason,
            "request_type": request_type,
        },
        headers=headers,
        content_type="application/json",
    )

    response = result.get_json()
    assert result.status_code == 200
    assert response["id"] == 1
    assert response["object_id"] == object_id
    assert response["reason"] == reason
    assert response["request_type"] == request_type
    assert response["status"] == "pending"


def test_get_specific_request(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    object_id = "61612325"
    reason = "this is a sample reason"
    request_type = "budget"

    create = client.post(
        "/dcfl/requests",
        json={
            "object_id": object_id,
            "reason": reason,
            "request_type": request_type,
        },
        headers=headers,
    )

    request_id = create.get_json()["id"]

    result = client.get(
        "/dcfl/requests/" + str(request_id),
        headers=headers,
        content_type="application/json",
    )
    response = result.get_json()

    assert create.status_code == 200
    assert result.status_code == 200
    assert response["id"] == request_id
    assert response["object_id"] == object_id
    assert response["reason"] == reason
    assert response["request_type"] == request_type
    assert response["status"] == "pending"


def test_get_all_requests(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    object_id = "61612325"
    reason = "sample reason"
    request_type = "permissions"

    create = client.post(
        "/dcfl/requests",
        json={
            "object_id": object_id,
            "reason": reason,
            "request_type": request_type,
        },
        headers=headers,
    )

    object_id = "61612325"
    reason = "sample reason"
    request_type = "permissions"

    result = client.get(
        "/dcfl/requests", headers=headers, content_type="application/json"
    )

    response = result.get_json()
    assert result.status_code == 200
    assert object_id in [el["object_id"] for el in response]
    assert reason in [el["reason"] for el in response]
    assert request_type in [el["request_type"] for el in response]


@pytest.mark.skip(reason="Should be made in integration tests")
def test_update_request(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    object_id = "61612325"
    reason = "this is a sample reason"
    request_type = "budget"

    create = client.post(
        "/dcfl/requests",
        json={
            "object_id": object_id,
            "reason": reason,
            "request_type": request_type,
        },
        headers=headers,
    )

    status = "accepted"
    request_id = "1"

    client.put(
        "/dcfl/requests/" + request_id,
        json={"status": status},
        headers=headers,
        content_type="application/json",
    )

    result = client.get(
        "/dcfl/requests/" + request_id,
        headers=headers,
        content_type="application/json",
    )

    response = result.get_json()
    assert result.status_code == 200
    assert response["id"] == int(request_id)
    # assert response["status"] == "accepted"


@pytest.mark.skip(reason="Should be made in integration tests")
def test_delete_request(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    object_id = "61612325"
    reason = "this is a sample reason"
    request_type = "budget"

    create = client.post(
        "/dcfl/requests",
        json={
            "object_id": object_id,
            "reason": reason,
            "request_type": request_type,
        },
        headers=headers,
    )

    request_id = "1"

    result = client.delete(
        "/dcfl/requests/" + request_id,
        headers=headers,
        content_type="application/json",
    )

    response = result.get_json()
    assert result.status_code == 200
    assert response["msg"] == "Request deleted!"
