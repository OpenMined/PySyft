# stdlib

# third party
import pytest

# syft absolute
from syft.server.credentials import SyftSigningKey
from syft.server.credentials import SyftVerifyKey
from syft.service.action.action_object import ActionObject
from syft.service.action.action_permissions import ActionObjectOWNER
from syft.service.action.action_permissions import ActionObjectPermission
from syft.service.action.action_store import ActionObjectEXECUTE
from syft.service.action.action_store import ActionObjectREAD
from syft.service.action.action_store import ActionObjectStash
from syft.service.action.action_store import ActionObjectWRITE
from syft.service.user.user import User
from syft.service.user.user_roles import ServiceRole
from syft.service.user.user_stash import UserStash
from syft.store.db.db import DBManager
from syft.types.uid import UID

# relative
from ..worker_test import action_object_stash  # noqa: F401

permissions = [
    ActionObjectOWNER,
    ActionObjectREAD,
    ActionObjectWRITE,
    ActionObjectEXECUTE,
]


def add_user(db_manager: DBManager, role: ServiceRole) -> SyftVerifyKey:
    user_stash = UserStash(store=db_manager)
    verify_key = SyftSigningKey.generate().verify_key
    user_stash.set(
        credentials=db_manager.root_verify_key,
        obj=User(verify_key=verify_key, role=role, id=UID()),
    ).unwrap()
    return verify_key


def add_test_object(
    stash: ActionObjectStash, verify_key: SyftVerifyKey
) -> ActionObject:
    test_object = ActionObject.from_obj([1, 2, 3])
    uid = test_object.id
    stash.set_or_update(
        uid=uid,
        credentials=verify_key,
        syft_object=test_object,
        has_result_read_permission=True,
    ).unwrap()
    return uid


@pytest.mark.parametrize(
    "stash",
    [
        pytest.lazy_fixture("action_object_stash"),
    ],
)
@pytest.mark.parametrize("permission", permissions)
def test_action_store_test_permissions(
    stash: ActionObjectStash, permission: ActionObjectPermission
) -> None:
    client_key = add_user(stash.db, ServiceRole.DATA_SCIENTIST)
    root_key = add_user(stash.db, ServiceRole.ADMIN)
    hacker_key = add_user(stash.db, ServiceRole.DATA_SCIENTIST)
    new_admin_key = add_user(stash.db, ServiceRole.ADMIN)

    test_item_id = add_test_object(stash, client_key)

    access = permission(uid=test_item_id, credentials=client_key)
    access_root = permission(uid=test_item_id, credentials=root_key)
    access_hacker = permission(uid=test_item_id, credentials=hacker_key)
    access_new_admin = permission(uid=test_item_id, credentials=new_admin_key)

    stash.add_permission(access)
    assert stash.has_permission(access)
    assert stash.has_permission(access_root)
    assert stash.has_permission(access_new_admin)
    assert not stash.has_permission(access_hacker)

    # remove permission
    stash.remove_permission(access)

    assert not stash.has_permission(access)
    assert stash.has_permission(access_root)
    assert stash.has_permission(access_new_admin)
    assert not stash.has_permission(access_hacker)

    # take ownership with new UID
    item2_id = add_test_object(stash, client_key)
    access = permission(uid=item2_id, credentials=client_key)

    stash.add_permission(ActionObjectREAD(uid=item2_id, credentials=client_key))
    assert stash.has_permission(access)
    assert stash.has_permission(access_root)
    assert stash.has_permission(access_new_admin)
    assert not stash.has_permission(access_hacker)

    # delete UID as hacker

    res = stash.delete_by_uid(hacker_key, item2_id)

    assert res.is_err()
    assert stash.has_permission(access)
    assert stash.has_permission(access_root)
    assert stash.has_permission(access_new_admin)
    assert not stash.has_permission(access_hacker)

    # delete UID as owner
    res = stash.delete_by_uid(client_key, item2_id)
    assert res.is_ok()
    assert not stash.has_permission(access)
    assert stash.has_permission(access_new_admin)
    assert not stash.has_permission(access_hacker)


@pytest.mark.parametrize(
    "stash",
    [
        pytest.lazy_fixture("action_object_stash"),
    ],
)
def test_action_store_test_dataset_get(stash: ActionObjectStash) -> None:
    client_key = add_user(stash.db, ServiceRole.DATA_SCIENTIST)
    root_key = add_user(stash.db, ServiceRole.ADMIN)

    data_uid = add_test_object(stash, client_key)
    access = ActionObjectWRITE(uid=data_uid, credentials=client_key)
    access_root = ActionObjectWRITE(uid=data_uid, credentials=root_key)
    read_permission = ActionObjectREAD(uid=data_uid, credentials=client_key)

    # add permission
    stash.add_permission(access)

    assert stash.has_permission(access)
    assert stash.has_permission(access_root)

    stash.add_permission(read_permission)
    assert stash.has_permission(read_permission)

    # check that trying to get action data that doesn't exist returns an error, even if have permissions
    stash.delete_by_uid(client_key, data_uid)
    res = stash.get(data_uid, client_key)
    assert res.is_err()
