# stdlib
import sys
from typing import Any

# third party
import pytest

# syft absolute
from syft.server.credentials import SyftVerifyKey
from syft.service.action.action_store import ActionObjectEXECUTE
from syft.service.action.action_store import ActionObjectOWNER
from syft.service.action.action_store import ActionObjectREAD
from syft.service.action.action_store import ActionObjectWRITE
from syft.types.uid import UID

# relative
from .store_constants_test import TEST_VERIFY_KEY_NEW_ADMIN
from .store_constants_test import TEST_VERIFY_KEY_STRING_CLIENT
from .store_constants_test import TEST_VERIFY_KEY_STRING_HACKER
from .store_constants_test import TEST_VERIFY_KEY_STRING_ROOT
from .store_mocks_test import MockSyftObject

permissions = [
    ActionObjectOWNER,
    ActionObjectREAD,
    ActionObjectWRITE,
    ActionObjectEXECUTE,
]


@pytest.mark.parametrize(
    "store",
    [
        pytest.lazy_fixture("dict_action_store"),
        pytest.lazy_fixture("sqlite_action_store"),
        pytest.lazy_fixture("mongo_action_store"),
    ],
)
def test_action_store_sanity(store: Any):
    assert hasattr(store, "store_config")
    assert hasattr(store, "settings")
    assert hasattr(store, "data")
    assert hasattr(store, "permissions")
    assert hasattr(store, "root_verify_key")
    assert store.root_verify_key.verify == TEST_VERIFY_KEY_STRING_ROOT


@pytest.mark.parametrize(
    "store",
    [
        pytest.lazy_fixture("dict_action_store"),
        pytest.lazy_fixture("sqlite_action_store"),
        pytest.lazy_fixture("mongo_action_store"),
    ],
)
@pytest.mark.parametrize("permission", permissions)
@pytest.mark.flaky(reruns=3, reruns_delay=3)
@pytest.mark.skipif(sys.platform == "darwin", reason="skip on mac")
def test_action_store_test_permissions(store: Any, permission: Any):
    client_key = SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_CLIENT)
    root_key = SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_ROOT)
    hacker_key = SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_HACKER)
    new_admin_key = TEST_VERIFY_KEY_NEW_ADMIN

    access = permission(uid=UID(), credentials=client_key)
    access_root = permission(uid=UID(), credentials=root_key)
    access_hacker = permission(uid=UID(), credentials=hacker_key)
    access_new_admin = permission(uid=UID(), credentials=new_admin_key)

    # add permission
    store.add_permission(access)

    assert store.has_permission(access)
    assert store.has_permission(access_root)
    assert store.has_permission(access_new_admin)
    assert not store.has_permission(access_hacker)

    # remove permission
    store.remove_permission(access)

    assert not store.has_permission(access)
    assert store.has_permission(access_root)
    assert store.has_permission(access_new_admin)
    assert not store.has_permission(access_hacker)

    # take ownership with new UID
    client_uid2 = UID()
    access = permission(uid=client_uid2, credentials=client_key)

    store.take_ownership(client_uid2, client_key)
    assert store.has_permission(access)
    assert store.has_permission(access_root)
    assert store.has_permission(access_new_admin)
    assert not store.has_permission(access_hacker)

    # delete UID as hacker
    access_hacker_ro = ActionObjectREAD(uid=UID(), credentials=hacker_key)
    store.add_permission(access_hacker_ro)

    res = store.delete(client_uid2, hacker_key)

    assert res.is_err()
    assert store.has_permission(access)
    assert store.has_permission(access_new_admin)
    assert store.has_permission(access_hacker_ro)

    # delete UID as owner
    res = store.delete(client_uid2, client_key)
    assert res.is_ok()
    assert not store.has_permission(access)
    assert store.has_permission(access_new_admin)
    assert not store.has_permission(access_hacker)


@pytest.mark.parametrize(
    "store",
    [
        pytest.lazy_fixture("dict_action_store"),
        pytest.lazy_fixture("sqlite_action_store"),
        pytest.lazy_fixture("mongo_action_store"),
    ],
)
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_action_store_test_dataset_get(store: Any):
    client_key = SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_CLIENT)
    root_key = SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_ROOT)
    SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_HACKER)

    permission_only_uid = UID()
    access = ActionObjectWRITE(uid=permission_only_uid, credentials=client_key)
    access_root = ActionObjectWRITE(uid=permission_only_uid, credentials=root_key)
    read_permission = ActionObjectREAD(uid=permission_only_uid, credentials=client_key)

    # add permission
    store.add_permission(access)

    assert store.has_permission(access)
    assert store.has_permission(access_root)

    store.add_permission(read_permission)
    assert store.has_permission(read_permission)

    # check that trying to get action data that doesn't exist returns an error, even if have permissions
    res = store.get(permission_only_uid, client_key)
    assert res.is_err()

    # add data
    data_uid = UID()
    obj = MockSyftObject(data=1)

    res = store.set(data_uid, client_key, obj, has_result_read_permission=True)
    assert res.is_ok()
    res = store.get(data_uid, client_key)
    assert res.is_ok()
    assert res.ok() == obj

    assert store.exists(data_uid)
    res = store.delete(data_uid, client_key)
    assert res.is_ok()
    res = store.delete(data_uid, client_key)
    assert res.is_err()
