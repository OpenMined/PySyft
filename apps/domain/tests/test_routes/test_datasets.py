from base64 import b64encode, b64decode
from json import dumps

from syft.core.store.storeable_object import StorableObject
from syft.core.store import Dataset
from syft.core.common import UID
from flask import current_app as app
import torch as th
import pytest
import jwt

from src.main.core.database.store_disk import (
    DiskObjectStore,
    create_dataset,
    create_storable,
)
from src.main.core.database.bin_storage.metadata import StorageMetadata, get_metadata
from src.main.core.database.bin_storage.bin_obj import BinaryObject
from src.main.core.database import *

ENCODING = "UTF-8"
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

storable = create_storable(
    _id=UID(),
    data=th.Tensor([1, 2, 3, 4]),
    description="Dummy tensor",
    tags=["dummy", "tensor"],
)
storable2 = create_storable(
    _id=UID(),
    data=th.Tensor([-1, -2, -3, -4]),
    description="Negative Dummy tensor",
    tags=["negative", "dummy", "tensor"],
)

storable3 = create_storable(
    _id=UID(),
    data=th.Tensor([11, 22, 33, 44]),
    description="NewDummy tensor",
    tags=["new", "dummy", "tensor"],
)

dataset = create_dataset(
    _id=UID(),
    data=[storable, storable2],
    description="Dummy tensor",
    tags=["dummy", "tensor"],
)


@pytest.fixture
def cleanup(database):
    yield
    try:
        database.session.query(User).delete()
        database.session.query(Role).delete()
        database.session.query(Group).delete()
        database.session.query(UserGroup).delete()
        database.session.query(BinaryObject).delete()
        database.session.query(StorageMetadata).delete()
        database.session.commit()
    except:
        database.session.rollback()


def test_create_dataset(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    obj_bytes = dataset.to_bytes()
    serialized = b64encode(obj_bytes)
    serialized = serialized.decode(ENCODING)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"dataset": serialized}
    result = client.post(
        "/dcfl/datasets",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 200

    _id = [v for v in result.get_json().keys()][0]
    retrieved = [v for v in result.get_json().values()][0]

    assert retrieved == serialized
    assert database.session.query(BinaryObject).get(_id).binary == obj_bytes


def test_get_all_datasets(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    uid1 = UID()
    new_dataset = create_dataset(
        _id=uid1,
        data=[storable2],
        description="Dummy tensor 1",
        tags=["dummy1", "tensor"],
    )
    storage = DiskObjectStore(database)
    obj1_bytes = dataset.to_bytes()
    obj2_bytes = new_dataset.to_bytes()
    _id1 = storage.store_bytes(obj1_bytes)
    _id2 = storage.store_bytes(obj2_bytes)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get(
        "/dcfl/datasets", headers=headers, content_type="application/json"
    )

    assert result.status_code == 200
    datasets = result.get_json().get("datasets", None)
    assert datasets is not None
    assert datasets.get(_id1, None) is not None
    assert datasets.get(_id2, None) is not None
    assert b64decode(datasets.get(_id1, None)) == obj1_bytes
    assert b64decode(datasets.get(_id2, None)) == obj2_bytes


def test_get_specific_dataset(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    obj_bytes = dataset.to_bytes()
    storage = DiskObjectStore(database)
    _id = storage.store_bytes(obj_bytes)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get(
        "/dcfl/datasets/{}".format(_id),
        headers=headers,
        content_type="application/json",
    )
    assert result.status_code == 200

    retrieved = result.get_json().get(_id, None)
    assert retrieved is not None

    retrieved = b64decode(retrieved)
    assert retrieved == obj_bytes


def test_update_dataset(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    uid1 = UID()
    new_dataset = create_dataset(
        _id=uid1,
        data=[storable2],
        description="Dummy tensor 1",
        tags=["dummy1", "tensor"],
    )
    storage = DiskObjectStore(database)
    obj1_bytes = dataset.to_bytes()
    obj2_bytes = new_dataset.to_bytes()
    _id = storage.store_bytes(obj1_bytes)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    assert database.session.query(BinaryObject).get(_id).binary == obj1_bytes

    result = client.put(
        "/dcfl/datasets/{}".format(_id),
        data=dumps({"dataset": b64encode(obj2_bytes).decode(ENCODING)}),
        headers=headers,
        content_type="application/json",
    )

    assert result.status_code == 204
    assert database.session.query(BinaryObject).get(_id).binary == obj2_bytes


def test_delete_dataset(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    storage = DiskObjectStore(database)
    obj_bytes = dataset.to_bytes()
    _id = storage.store_bytes(obj_bytes)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    assert database.session.query(BinaryObject).get(_id).binary == obj_bytes

    result = client.delete(
        "/dcfl/datasets/{}".format(_id),
        headers=headers,
        content_type="application/json",
    )

    assert result.status_code == 204
