# stdlib
from pathlib import Path
from typing import Any
from typing import Generator

# third party
import pytest

# syft absolute
from syft.core.node.new.action_store import ActionObjectEXECUTE
from syft.core.node.new.action_store import ActionObjectOWNER
from syft.core.node.new.action_store import ActionObjectREAD
from syft.core.node.new.action_store import ActionObjectWRITE
from syft.core.node.new.action_store import DictActionStore
from syft.core.node.new.action_store import SQLiteActionStore
from syft.core.node.new.credentials import SyftVerifyKey
from syft.core.node.new.dict_document_store import DictStoreConfig
from syft.core.node.new.sqlite_document_store import SQLiteStoreClientConfig
from syft.core.node.new.sqlite_document_store import SQLiteStoreConfig
from syft.core.node.new.uid import UID

# relative
from .store_mocks_test import MockSyftObject

workspace = Path("workspace")
db_name = "testing"

permissions = [
    ActionObjectOWNER,
    ActionObjectREAD,
    ActionObjectWRITE,
    ActionObjectEXECUTE,
]

test_verify_key_string_root = (
    "08e5bcddfd55cdff0f7f6a62d63a43585734c6e7a17b2ffb3f3efe322c3cecc5"
)
test_verify_key_string_client = (
    "833035a1c408e7f2176a0b0cd4ba0bc74da466456ea84f7ba4e28236e7e303ab"
)
test_verify_key_string_hacker = (
    "8f4412396d3418d17c08a8f46592621a5d57e0daf1c93e2134c30f50d666801d"
)


@pytest.fixture(autouse=True)
def prepare_workspace() -> Generator:
    workspace.mkdir(parents=True, exist_ok=True)
    db_path = workspace / db_name

    if db_path.exists():
        db_path.unlink()

    yield

    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def dict_store():
    store_config = DictStoreConfig()
    ver_key = SyftVerifyKey.from_string(test_verify_key_string_root)
    return DictActionStore(store_config=store_config, root_verify_key=ver_key)


@pytest.fixture
def sqlite_store():
    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    ver_key = SyftVerifyKey.from_string(test_verify_key_string_root)
    return SQLiteActionStore(store_config=store_config, root_verify_key=ver_key)


@pytest.mark.parametrize(
    "store",
    [
        pytest.lazy_fixture("dict_store"),
        pytest.lazy_fixture("sqlite_store"),
    ],
)
def test_action_store_sanity(store: Any):
    assert hasattr(store, "store_config")
    assert hasattr(store, "settings")
    assert hasattr(store, "data")
    assert hasattr(store, "permissions")
    assert hasattr(store, "root_verify_key")
    assert store.root_verify_key.verify == test_verify_key_string_root


@pytest.mark.parametrize(
    "store",
    [
        pytest.lazy_fixture("dict_store"),
        pytest.lazy_fixture("sqlite_store"),
    ],
)
@pytest.mark.parametrize("permission", permissions)
def test_action_store_test_permissions(store: Any, permission: Any):
    client_key = SyftVerifyKey.from_string(test_verify_key_string_client)
    root_key = SyftVerifyKey.from_string(test_verify_key_string_root)
    hacker_key = SyftVerifyKey.from_string(test_verify_key_string_hacker)

    access = permission(uid=UID(), credentials=client_key)
    access_root = permission(uid=UID(), credentials=root_key)
    access_hacker = permission(uid=UID(), credentials=hacker_key)

    # add permission
    store.add_permission(access)

    assert store.has_permission(access)
    assert store.has_permission(access_root)
    assert not store.has_permission(access_hacker)

    # remove permission
    store.remove_permission(access)

    assert not store.has_permission(access)
    assert store.has_permission(access_root)
    assert not store.has_permission(access_hacker)

    # take ownership with new UID
    client_uid2 = UID()
    access = permission(uid=client_uid2, credentials=client_key)

    store.take_ownership(client_uid2, client_key)
    assert store.has_permission(access)
    assert store.has_permission(access_root)
    assert not store.has_permission(access_hacker)

    # delete UID as hacker
    access_hacker_ro = ActionObjectREAD(uid=UID(), credentials=hacker_key)
    store.add_permission(access_hacker_ro)

    res = store.delete(client_uid2, hacker_key)

    assert res.is_err()
    assert store.has_permission(access)
    assert store.has_permission(access_hacker_ro)

    # delete UID as owner
    res = store.delete(client_uid2, client_key)
    assert res.is_ok()
    assert not store.has_permission(access)
    assert not store.has_permission(access_hacker)


@pytest.mark.parametrize(
    "store",
    [
        pytest.lazy_fixture("dict_store"),
        pytest.lazy_fixture("sqlite_store"),
    ],
)
def test_action_store_test_data_set_get(store: Any):
    client_key = SyftVerifyKey.from_string(test_verify_key_string_client)
    root_key = SyftVerifyKey.from_string(test_verify_key_string_root)
    SyftVerifyKey.from_string(test_verify_key_string_hacker)

    access = ActionObjectWRITE(uid=UID(), credentials=client_key)
    access_root = ActionObjectWRITE(uid=UID(), credentials=root_key)

    # add permission
    store.add_permission(access)

    assert store.has_permission(access)
    assert store.has_permission(access_root)

    # add data
    data_uid = UID()
    obj = MockSyftObject(data=1)

    res = store.set(data_uid, client_key, obj)
    assert res.is_ok()
    res = store.get(data_uid, client_key)
    assert res.is_ok()
    assert res.ok() == obj

    assert store.exists(data_uid)
    res = store.delete(data_uid, client_key)
    assert res.is_ok()
    res = store.delete(data_uid, client_key)
    assert res.is_err()
