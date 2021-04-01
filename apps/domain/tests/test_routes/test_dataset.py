from base64 import b64encode, b64decode
from json import dumps

from syft import deserialize
from syft.core.store.storeable_object import StorableObject
from syft.core.store import Dataset
from syft.core.common import UID
from flask import current_app as app
import torch as th
import pytest
import jwt
import os

from src.main.core.database.store_disk import (
    DiskObjectStore,
    create_storable,
)
from src.main.core.database.bin_storage.metadata import StorageMetadata, get_metadata
from src.main.core.database.bin_storage.bin_obj import BinObject
from src.main.core.datasets.dataset_ops import create_dataset

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

tensor3 = {
    "content": "11, 22, 33, 44\n111, 222, 333, 444",
    "manifest": "Aenean at dictum ipsum",
    "description": "NewDummy tensor",
    "tags": ["new", "dummy", "tensor"],
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
        database.session.query(BinObject).delete()
        database.session.query(JsonObject).delete()
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

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    folder = os.path.dirname(__file__)
    file1 = open(f"{folder}/mtcars_train.csv", "rb")
    file1 = file1.read().decode("utf-8")
    file2 = open(f"{folder}/mtcars_test.csv", "rb")
    file2 = file2.read().decode("utf-8")

    payload = {
        "name": "Cars dataset",
        "description": " ... ",
        "manifest": "Columns: mpg,cyl,disp,hp,drat,wt,qsec,vs,am,gear,carb",
        "tags": ["#hashtag", "#diabetes"],
        "created_at": "05/12/2020",
        "tensors": {
            "train": {"content": file1, "manifest": ""},
            "test": {"content": file2, "manifest": ""},
        },
    }

    result = client.post(
        "/data-centric/datasets",
        headers=headers,
        data=dumps(payload),
        content_type="multipart/form-data",
    )

    assert result.status_code == 200

    _id = result.get_json().get("id", None)
    storables = (
        database.session.query(DatasetGroup.bin_object).filter_by(dataset=_id).all()
    )
    assert storables is not None
    assert len(storables) == 2
    storables = [el[0] for el in storables]

    assert database.session.query(BinObject).get(storables[0]) is not None
    assert database.session.query(BinObject).get(storables[1]) is not None

    assert database.session.query(JsonObject).get(_id) is not None
    assert database.session.query(JsonObject).get(_id).binary is not None

    _json = database.session.query(JsonObject).get(_id).binary
    assert _json["id"] == _id
    assert _json["tags"] == payload["tags"]
    assert _json["manifest"] == payload["manifest"]
    assert _json["created_at"] == payload["created_at"]


def test_get_all_datasets_metadata(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    new_dataset = {
        "name": "Dummy Dataset 1",
        "description": "Lorem ipsum dolor",
        "manifest": "Etiam vestibulum velit a tellus aliquet varius",
        "tags": ["#hashtag", "#dummy"],
        "created_at": "05/12/2019",
        "tensors": {"train": tensor2.copy()},
    }
    storage = DiskObjectStore(database)
    df_json1 = create_dataset(dataset)
    df_json2 = create_dataset(new_dataset)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get(
        "/data-centric/datasets", headers=headers, content_type="application/json"
    )

    assert result.status_code == 200

    assert df_json1["id"] in [el["id"] for el in result.get_json()]
    assert df_json1["description"] in [el["description"] for el in result.get_json()]
    assert df_json1["manifest"] in [el["manifest"] for el in result.get_json()]

    assert df_json2["id"] in [el["id"] for el in result.get_json()]
    assert df_json2["description"] in [el["description"] for el in result.get_json()]
    assert df_json2["manifest"] in [el["manifest"] for el in result.get_json()]


def test_get_specific_dataset_metadata(client, database, cleanup):
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
    df_metadata = create_dataset(dataset)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get(
        "/data-centric/datasets/{}".format(df_metadata["id"]),
        headers=headers,
        content_type="application/json",
    )

    assert result.status_code == 200
    assert result.get_json()["id"] == df_metadata["id"]
    assert result.get_json()["tags"] == df_metadata["tags"]
    assert result.get_json()["name"] == df_metadata["name"]
    assert result.get_json()["manifest"] == df_metadata["manifest"]
    assert result.get_json()["description"] == df_metadata["description"]


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

    new_dataset = {
        "name": "Dummy Dataset 1",
        "description": "Lorem ipsum dolor",
        "manifest": "Etiam vestibulum velit a tellus aliquet varius",
        "tags": ["#tensor", "#dummy1"],
        "created_at": "19/06/1972",
        "tensors": {"train": tensor2.copy()},
    }
    storage = DiskObjectStore(database)
    df_json1 = create_dataset(dataset)

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    assert (
        database.session.query(DatasetGroup).filter_by(dataset=df_json1["id"]).all()
        is not None
    )
    assert database.session.query(JsonObject).get(df_json1["id"]) is not None
    assert database.session.query(JsonObject).get(df_json1["id"]).binary == df_json1

    result = client.put(
        "/data-centric/datasets/{}".format(df_json1["id"]),
        data=dumps(new_dataset),
        headers=headers,
        content_type="application/json",
    )

    assert result.status_code == 200
    assert result.get_json()["id"] == df_json1["id"]

    assert (
        database.session.query(DatasetGroup).filter_by(dataset=df_json1["id"]).all()
        is not None
    )
    assert (
        len(
            database.session.query(DatasetGroup).filter_by(dataset=df_json1["id"]).all()
        )
        == 1
    )
    assert database.session.query(JsonObject).get(df_json1["id"]) is not None

    metadata = database.session.query(JsonObject).get(df_json1["id"])
    assert metadata is not None
    metadata = metadata.binary

    assert metadata["description"] == new_dataset["description"]
    assert metadata["manifest"] == new_dataset["manifest"]
    assert metadata["created_at"] == new_dataset["created_at"]
    assert metadata["tags"] == new_dataset["tags"]
    assert metadata["name"] == new_dataset["name"]


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
    df_json1 = create_dataset(dataset)
    _id = df_json1["id"]

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    assert database.session.query(DatasetGroup).filter_by(dataset=_id).all() is not None

    storable_ids = (
        database.session.query(DatasetGroup.bin_object).filter_by(dataset=_id).all()
    )
    storable_ids = [x[0] for x in storable_ids]

    assert database.session.query(JsonObject).get(_id) is not None
    assert database.session.query(JsonObject).get(_id).binary is not None
    assert (
        database.session.query(JsonObject).get(_id).binary["description"]
        == dataset["description"]
    )
    assert database.session.query(JsonObject).get(_id).binary["tags"] == dataset["tags"]

    result = client.delete(
        "/data-centric/datasets/{}".format(_id),
        headers=headers,
        content_type="application/json",
    )

    assert result.status_code == 204

    for strbl_id in storable_ids:
        assert db.session.query(BinObject).filter_by(id=strbl_id).first() is None
        assert db.session.query(ObjectMetadata).filter_by(id=strbl_id).first() is None

    assert database.session.query(DatasetGroup).filter_by(dataset=_id).all() == []
    assert database.session.query(JsonObject).get(_id) is None
