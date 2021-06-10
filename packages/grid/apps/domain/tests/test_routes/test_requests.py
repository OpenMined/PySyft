# stdlib
from json import dumps
from json import loads
import time

# third party
from flask import current_app as app
import jwt
import pytest
from src.main.core.database import *
from src.main.core.database.store_disk import DiskObjectStore
from src.main.core.datasets.dataset_ops import create_dataset

owner_role = ("Owner", True, True, True, True, True, True, True)

user1 = (
    "tech@gibberish.com",
    "BDEB6E8EE39B6C70835993486C9E65DC",
    "]GBF[R>GX[9Cmk@DthFT!mhloUc%[f",
    "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
    1,
)

tensor1 = {
    "content": "1, 2, 3, 4\n10, 20, 30, 40",
    "manifest": "Suspendisse et fermentum lectus",
    "description": "Dummy tensor",
    "tags": ["dummy", "tensor"],
}

tensor2 = {
    "content": "-1, -2, -3, -4,\n-100, -200, -300, -400",
    "manifest": "Suspendisse et fermentum lectus",
    "description": "Negative Dummy tensor",
    "tags": ["negative", "dummy", "tensor"],
}

dataset = {
    "name": "Dummy Dataset",
    "description": "Neque porro quisquam",
    "manifest": "Sed vehicula mauris non turpis sollicitudin congue.",
    "tags": ["#hashtag", "#dummy", "#original"],
    "created_at": "05/12/2018",
    "tensors": {"train": tensor1.copy(), "test": tensor2.copy()},
}


@pytest.fixture
def cleanup(database):
    yield
    try:
        database.session.query(User).delete()
        database.session.query(Role).delete()
        database.session.query(Group).delete()
        database.session.query(UserGroup).delete()
        database.session.query(Request).delete()
        database.session.query(DatasetGroup).delete()
        database.session.query(BinObject).delete()
        database.session.query(JsonObject).delete()
        database.session.query(StorageMetadata).delete()

        database.session.commit()
    except:
        database.session.rollback()


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

    storage = DiskObjectStore(database)
    dataset_json = create_dataset(dataset)

    object_id = dataset_json["tensors"]["train"]["id"]
    reason = "sample reason"
    request_type = "permissions"

    result = client.post(
        "/data-centric/requests",
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
    assert response["object_id"] == object_id
    assert response["reason"] == reason
    assert response["request_type"] == request_type
    assert response["status"] == "pending"


def test_create_duplicate_fail(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()
    storage = DiskObjectStore(database)
    dataset_json = create_dataset(dataset)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    object_id = dataset_json["tensors"]["train"]["id"]
    reason = "sample reason"
    request_type = "permissions"

    result1 = client.post(
        "/data-centric/requests",
        json={
            "object_id": object_id,
            "reason": reason,
            "request_type": request_type,
        },
        headers=headers,
    )

    result1 = client.get(
        "/data-centric/requests", headers=headers, content_type="application/json"
    )

    result2 = client.post(
        "/data-centric/requests",
        json={
            "object_id": object_id,
            "reason": reason,
            "request_type": request_type,
        },
        headers=headers,
    )

    assert result1.status_code == 200
    assert object_id in [el["object_id"] for el in result1.get_json()]
    assert reason in [el["reason"] for el in result1.get_json()]
    assert request_type in [el["request_type"] for el in result1.get_json()]

    assert result2.status_code == 403


def test_get_specific_request(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()
    storage = DiskObjectStore(database)
    dataset_json = create_dataset(dataset)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    object_id = dataset_json["tensors"]["train"]["id"]
    reason = "this is a sample reason"
    request_type = "budget"

    create = client.post(
        "/data-centric/requests",
        json={
            "object_id": object_id,
            "reason": reason,
            "request_type": request_type,
        },
        headers=headers,
    )

    request_id = create.get_json()["id"]

    result = client.get(
        "/data-centric/requests/" + str(request_id),
        headers=headers,
        content_type="application/json",
    )

    assert create.status_code == 200
    assert result.status_code == 200
    assert result.get_json()["id"] == request_id
    assert result.get_json()["object_id"] == object_id
    assert result.get_json()["reason"] == reason
    assert result.get_json()["request_type"] == request_type
    assert result.get_json()["status"] == "pending"


def test_get_all_requests(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()
    storage = DiskObjectStore(database)
    dataset_json = create_dataset(dataset)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    object_id = dataset_json["tensors"]["train"]["id"]
    reason = "sample reason"
    request_type = "permissions"

    create = client.post(
        "/data-centric/requests",
        json={
            "object_id": object_id,
            "reason": reason,
            "request_type": request_type,
        },
        headers=headers,
    )

    result = client.get(
        "/data-centric/requests", headers=headers, content_type="application/json"
    )

    response = result.get_json()
    assert result.status_code == 200
    assert object_id in [el["object_id"] for el in response]
    assert reason in [el["reason"] for el in response]
    assert request_type in [el["request_type"] for el in response]


def test_update_request(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()
    storage = DiskObjectStore(database)
    dataset_json = create_dataset(dataset)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    object_id = dataset_json["tensors"]["train"]["id"]
    reason = "this is a sample reason"
    request_type = "budget"

    create = client.post(
        "/data-centric/requests",
        json={
            "object_id": object_id,
            "reason": reason,
            "request_type": request_type,
        },
        headers=headers,
    )

    status = "accepted"
    request_id = create.get_json()["id"]

    client.put(
        "/data-centric/requests/" + request_id,
        json={"status": status},
        headers=headers,
        content_type="application/json",
    )

    result = client.get(
        "/data-centric/requests/" + request_id,
        headers=headers,
        content_type="application/json",
    )

    response = result.get_json()
    assert result.status_code == 200
    assert response["id"] == request_id
    assert response["status"] == "accepted"


def test_delete_request(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()
    storage = DiskObjectStore(database)
    dataset_json = create_dataset(dataset)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    object_id = dataset_json["tensors"]["train"]["id"]
    reason = "this is a sample reason"
    request_type = "budget"

    create = client.post(
        "/data-centric/requests",
        json={
            "object_id": object_id,
            "reason": reason,
            "request_type": request_type,
        },
        headers=headers,
    )

    request_id = create.get_json()["id"]

    result = client.delete(
        "/data-centric/requests/" + request_id,
        headers=headers,
        content_type="application/json",
    )

    assert result.status_code == 204
