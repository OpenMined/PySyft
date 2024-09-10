# stdlib

# third party
import pytest

# syft absolute
from syft.server.credentials import SyftVerifyKey
from syft.service.action.action_permissions import ActionObjectOWNER
from syft.service.action.action_permissions import ActionObjectPermission
from syft.service.action.action_permissions import ActionPermission
from syft.service.action.action_permissions import StoragePermission
from syft.service.action.action_store import ActionObjectEXECUTE
from syft.service.action.action_store import ActionObjectREAD
from syft.service.action.action_store import ActionObjectWRITE
from syft.store.document_store import QueryKey
from syft.store.mongo_document_store import MongoStorePartition
from syft.types.errors import SyftException
from syft.types.uid import UID

# relative
from ...mongomock.collection import Collection as MongoCollection
from .store_constants_test import TEST_VERIFY_KEY_STRING_HACKER
from .store_mocks_test import MockSyftObject

PERMISSIONS = [
    ActionObjectOWNER,
    ActionObjectREAD,
    ActionObjectWRITE,
    ActionObjectEXECUTE,
]


def test_mongo_store_partition_add_remove_permission(
    root_verify_key: SyftVerifyKey, mongo_store_partition: MongoStorePartition
) -> None:
    """
    Test the add_permission and remove_permission functions of MongoStorePartition
    """
    # setting up
    permissions_collection: MongoCollection = mongo_store_partition.permissions.ok()
    obj = MockSyftObject(data=1)

    # add the first permission
    obj_read_permission = ActionObjectPermission(
        uid=obj.id, permission=ActionPermission.READ, credentials=root_verify_key
    )
    mongo_store_partition.add_permission(obj_read_permission)
    find_res_1 = permissions_collection.find_one({"_id": obj_read_permission.uid})
    assert find_res_1 is not None
    assert len(find_res_1["permissions"]) == 1
    assert find_res_1["permissions"] == {
        obj_read_permission.permission_string,
    }

    # add the second permission
    obj_write_permission = ActionObjectPermission(
        uid=obj.id, permission=ActionPermission.WRITE, credentials=root_verify_key
    )
    mongo_store_partition.add_permission(obj_write_permission)

    find_res_2 = permissions_collection.find_one({"_id": obj.id})
    assert find_res_2 is not None
    assert len(find_res_2["permissions"]) == 2
    assert find_res_2["permissions"] == {
        obj_read_permission.permission_string,
        obj_write_permission.permission_string,
    }

    # add duplicated permission
    mongo_store_partition.add_permission(obj_write_permission)
    find_res_3 = permissions_collection.find_one({"_id": obj.id})
    assert len(find_res_3["permissions"]) == 2
    assert find_res_3["permissions"] == find_res_2["permissions"]

    # remove the write permission
    mongo_store_partition.remove_permission(obj_write_permission)
    find_res_4 = permissions_collection.find_one({"_id": obj.id})
    assert len(find_res_4["permissions"]) == 1
    assert find_res_1["permissions"] == {
        obj_read_permission.permission_string,
    }

    # remove a non-existent permission
    with pytest.raises(SyftException):
        mongo_store_partition.remove_permission(
            ActionObjectPermission(
                uid=obj.id,
                permission=ActionPermission.OWNER,
                credentials=root_verify_key,
            )
        )
    find_res_5 = permissions_collection.find_one({"_id": obj.id})
    assert len(find_res_5["permissions"]) == 1
    assert find_res_1["permissions"] == {
        obj_read_permission.permission_string,
    }

    # there is only one permission object
    assert permissions_collection.count_documents({}) == 1

    # add permissions in a loop
    new_permissions = []
    repeats = 5
    for idx in range(1, repeats + 1):
        new_obj = MockSyftObject(data=idx)
        new_obj_read_permission = ActionObjectPermission(
            uid=new_obj.id,
            permission=ActionPermission.READ,
            credentials=root_verify_key,
        )
        new_permissions.append(new_obj_read_permission)
        mongo_store_partition.add_permission(new_obj_read_permission)
        assert permissions_collection.count_documents({}) == 1 + idx

    # remove all the permissions added in the loop
    for permission in new_permissions:
        mongo_store_partition.remove_permission(permission)

    assert permissions_collection.count_documents({}) == 1


def test_mongo_store_partition_add_remove_storage_permission(
    root_verify_key: SyftVerifyKey,
    mongo_store_partition: MongoStorePartition,
) -> None:
    """
    Test the add_storage_permission and remove_storage_permission functions of MongoStorePartition
    """

    obj = MockSyftObject(data=1)

    storage_permission = StoragePermission(
        uid=obj.id,
        server_uid=UID(),
    )
    assert not mongo_store_partition.has_storage_permission(storage_permission)
    mongo_store_partition.add_storage_permission(storage_permission)
    assert mongo_store_partition.has_storage_permission(storage_permission)
    mongo_store_partition.remove_storage_permission(storage_permission)
    assert not mongo_store_partition.has_storage_permission(storage_permission)

    obj2 = MockSyftObject(data=1)
    mongo_store_partition.set(root_verify_key, obj2, add_storage_permission=False)
    storage_permission3 = StoragePermission(
        uid=obj2.id, server_uid=mongo_store_partition.server_uid
    )
    assert not mongo_store_partition.has_storage_permission(storage_permission3)

    obj3 = MockSyftObject(data=1)
    mongo_store_partition.set(root_verify_key, obj3, add_storage_permission=True)
    storage_permission4 = StoragePermission(
        uid=obj3.id, server_uid=mongo_store_partition.server_uid
    )
    assert mongo_store_partition.has_storage_permission(storage_permission4)


def test_mongo_store_partition_add_permissions(
    root_verify_key: SyftVerifyKey,
    guest_verify_key: SyftVerifyKey,
    mongo_store_partition: MongoStorePartition,
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()
    permissions_collection: MongoCollection = mongo_store_partition.permissions.ok()
    obj = MockSyftObject(data=1)

    # add multiple permissions for the first object
    permission_1 = ActionObjectPermission(
        uid=obj.id, permission=ActionPermission.WRITE, credentials=root_verify_key
    )
    permission_2 = ActionObjectPermission(
        uid=obj.id, permission=ActionPermission.OWNER, credentials=root_verify_key
    )
    permission_3 = ActionObjectPermission(
        uid=obj.id, permission=ActionPermission.READ, credentials=guest_verify_key
    )
    permissions: list[ActionObjectPermission] = [
        permission_1,
        permission_2,
        permission_3,
    ]
    mongo_store_partition.add_permissions(permissions)

    # check if the permissions have been added properly
    assert permissions_collection.count_documents({}) == 1
    find_res = permissions_collection.find_one({"_id": obj.id})
    assert find_res is not None
    assert len(find_res["permissions"]) == 3

    # add permissions for the second object
    obj_2 = MockSyftObject(data=2)
    permission_4 = ActionObjectPermission(
        uid=obj_2.id, permission=ActionPermission.READ, credentials=root_verify_key
    )
    permission_5 = ActionObjectPermission(
        uid=obj_2.id, permission=ActionPermission.WRITE, credentials=root_verify_key
    )
    mongo_store_partition.add_permissions([permission_4, permission_5])

    assert permissions_collection.count_documents({}) == 2
    find_res_2 = permissions_collection.find_one({"_id": obj_2.id})
    assert find_res_2 is not None
    assert len(find_res_2["permissions"]) == 2


@pytest.mark.parametrize("permission", PERMISSIONS)
def test_mongo_store_partition_has_permission(
    root_verify_key: SyftVerifyKey,
    guest_verify_key: SyftVerifyKey,
    mongo_store_partition: MongoStorePartition,
    permission: ActionObjectPermission,
) -> None:
    hacker_verify_key = SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_HACKER)

    res = mongo_store_partition.init_store()
    assert res.is_ok()

    # root permission
    obj = MockSyftObject(data=1)
    permission_root = permission(uid=obj.id, credentials=root_verify_key)
    permission_client = permission(uid=obj.id, credentials=guest_verify_key)
    permission_hacker = permission(uid=obj.id, credentials=hacker_verify_key)
    mongo_store_partition.add_permission(permission_root)
    # only the root user has access to this permission
    assert mongo_store_partition.has_permission(permission_root)
    assert not mongo_store_partition.has_permission(permission_client)
    assert not mongo_store_partition.has_permission(permission_hacker)

    # client permission for another object
    obj_2 = MockSyftObject(data=2)
    permission_client_2 = permission(uid=obj_2.id, credentials=guest_verify_key)
    permission_root_2 = permission(uid=obj_2.id, credentials=root_verify_key)
    permisson_hacker_2 = permission(uid=obj_2.id, credentials=hacker_verify_key)
    mongo_store_partition.add_permission(permission_client_2)
    # the root (admin) and guest client should have this permission
    assert mongo_store_partition.has_permission(permission_root_2)
    assert mongo_store_partition.has_permission(permission_client_2)
    assert not mongo_store_partition.has_permission(permisson_hacker_2)

    # remove permissions
    mongo_store_partition.remove_permission(permission_root)
    assert not mongo_store_partition.has_permission(permission_root)
    assert not mongo_store_partition.has_permission(permission_client)
    assert not mongo_store_partition.has_permission(permission_hacker)

    mongo_store_partition.remove_permission(permission_client_2)
    assert not mongo_store_partition.has_permission(permission_root_2)
    assert not mongo_store_partition.has_permission(permission_client_2)
    assert not mongo_store_partition.has_permission(permisson_hacker_2)


def test_mongo_store_partition_permissions_set(
    root_verify_key: SyftVerifyKey,
    guest_verify_key: SyftVerifyKey,
    mongo_store_partition: MongoStorePartition,
) -> None:
    """
    Test the permissions functionalities when using MongoStorePartition._set function
    """
    hacker_verify_key = SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_HACKER)
    res = mongo_store_partition.init_store()
    assert res.is_ok()

    # set the object to mongo_store_partition.collection
    obj = MockSyftObject(data=1)
    res = mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
    assert res.is_ok()
    assert res.ok() == obj

    # check if the corresponding permissions has been added to the permissions
    # collection after the root client claim it
    pemissions_collection = mongo_store_partition.permissions.ok()
    assert isinstance(pemissions_collection, MongoCollection)
    permissions = pemissions_collection.find_one({"_id": obj.id})
    assert permissions is not None
    assert isinstance(permissions["permissions"], set)
    assert len(permissions["permissions"]) == 4
    for permission in PERMISSIONS:
        assert mongo_store_partition.has_permission(
            permission(uid=obj.id, credentials=root_verify_key)
        )

    # the hacker tries to set duplicated object but should not be able to claim it
    res_2 = mongo_store_partition.set(guest_verify_key, obj, ignore_duplicates=True)
    assert res_2.is_ok()
    for permission in PERMISSIONS:
        assert not mongo_store_partition.has_permission(
            permission(uid=obj.id, credentials=hacker_verify_key)
        )
        assert mongo_store_partition.has_permission(
            permission(uid=obj.id, credentials=root_verify_key)
        )


def test_mongo_store_partition_permissions_get_all(
    root_verify_key: SyftVerifyKey,
    guest_verify_key: SyftVerifyKey,
    mongo_store_partition: MongoStorePartition,
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()
    hacker_verify_key = SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_HACKER)
    # set several objects for the root and guest client
    num_root_objects: int = 5
    num_guest_objects: int = 3
    for i in range(num_root_objects):
        obj = MockSyftObject(data=i)
        mongo_store_partition.set(
            credentials=root_verify_key, obj=obj, ignore_duplicates=False
        )
    for i in range(num_guest_objects):
        obj = MockSyftObject(data=i)
        mongo_store_partition.set(
            credentials=guest_verify_key, obj=obj, ignore_duplicates=False
        )

    assert (
        len(mongo_store_partition.all(root_verify_key).ok())
        == num_root_objects + num_guest_objects
    )
    assert len(mongo_store_partition.all(guest_verify_key).ok()) == num_guest_objects
    assert len(mongo_store_partition.all(hacker_verify_key).ok()) == 0


def test_mongo_store_partition_permissions_delete(
    root_verify_key: SyftVerifyKey,
    guest_verify_key: SyftVerifyKey,
    mongo_store_partition: MongoStorePartition,
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()
    collection: MongoCollection = mongo_store_partition.collection.ok()
    pemissions_collection: MongoCollection = mongo_store_partition.permissions.ok()
    hacker_verify_key = SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_HACKER)

    # the root client set an object
    obj = MockSyftObject(data=1)
    mongo_store_partition.set(
        credentials=root_verify_key, obj=obj, ignore_duplicates=False
    )
    qk: QueryKey = mongo_store_partition.settings.store_key.with_obj(obj)
    # guest or hacker can't delete it
    assert not mongo_store_partition.delete(guest_verify_key, qk).is_ok()
    assert not mongo_store_partition.delete(hacker_verify_key, qk).is_ok()
    # only the root client can delete it
    assert mongo_store_partition.delete(root_verify_key, qk).is_ok()
    # check if the object and its permission have been deleted
    assert collection.count_documents({}) == 0
    assert pemissions_collection.count_documents({}) == 0

    # the guest client set an object
    obj_2 = MockSyftObject(data=2)
    mongo_store_partition.set(
        credentials=guest_verify_key, obj=obj_2, ignore_duplicates=False
    )
    qk_2: QueryKey = mongo_store_partition.settings.store_key.with_obj(obj_2)
    # the hacker can't delete it
    assert not mongo_store_partition.delete(hacker_verify_key, qk_2).is_ok()
    # the guest client can delete it
    assert mongo_store_partition.delete(guest_verify_key, qk_2).is_ok()
    assert collection.count_documents({}) == 0
    assert pemissions_collection.count_documents({}) == 0

    # the guest client set another object
    obj_3 = MockSyftObject(data=3)
    mongo_store_partition.set(
        credentials=guest_verify_key, obj=obj_3, ignore_duplicates=False
    )
    qk_3: QueryKey = mongo_store_partition.settings.store_key.with_obj(obj_3)
    # the root client also has the permission to delete it
    assert mongo_store_partition.delete(root_verify_key, qk_3).is_ok()
    assert collection.count_documents({}) == 0
    assert pemissions_collection.count_documents({}) == 0


def test_mongo_store_partition_permissions_update(
    root_verify_key: SyftVerifyKey,
    guest_verify_key: SyftVerifyKey,
    mongo_store_partition: MongoStorePartition,
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()
    # the root client set an object
    obj = MockSyftObject(data=1)
    mongo_store_partition.set(
        credentials=root_verify_key, obj=obj, ignore_duplicates=False
    )
    assert len(mongo_store_partition.all(credentials=root_verify_key).ok()) == 1

    qk: QueryKey = mongo_store_partition.settings.store_key.with_obj(obj)
    permsissions: MongoCollection = mongo_store_partition.permissions.ok()
    repeats = 5

    for v in range(repeats):
        # the guest client should not have permission to update obj
        obj_new = MockSyftObject(data=v)
        res = mongo_store_partition.update(
            credentials=guest_verify_key, qk=qk, obj=obj_new
        )
        assert res.is_err()
        # the root client has the permission to update obj
        res = mongo_store_partition.update(
            credentials=root_verify_key, qk=qk, obj=obj_new
        )
        assert res.is_ok()
        # the id of the object in the permission collection should not be changed
        assert permsissions.find_one(qk.as_dict_mongo)["_id"] == obj.id
