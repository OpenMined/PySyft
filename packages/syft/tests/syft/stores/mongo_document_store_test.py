# stdlib
import sys
from threading import Thread
from typing import List
from typing import Set
from typing import Tuple

# third party
from joblib import Parallel
from joblib import delayed
from pymongo.collection import Collection as MongoCollection
import pytest
from result import Err

# syft absolute
from syft.node.credentials import SyftVerifyKey
from syft.service.action.action_permissions import ActionObjectPermission
from syft.service.action.action_permissions import ActionPermission
from syft.service.action.action_store import ActionObjectEXECUTE
from syft.service.action.action_store import ActionObjectOWNER
from syft.service.action.action_store import ActionObjectREAD
from syft.service.action.action_store import ActionObjectWRITE
from syft.store.document_store import PartitionSettings
from syft.store.document_store import QueryKey
from syft.store.document_store import QueryKeys
from syft.store.mongo_client import MongoStoreClientConfig
from syft.store.mongo_document_store import MongoStoreConfig
from syft.store.mongo_document_store import MongoStorePartition

# relative
from .store_constants_test import generate_db_name
from .store_constants_test import test_verify_key_string_hacker
from .store_fixtures_test import mongo_store_partition_fn
from .store_mocks_test import MockObjectType
from .store_mocks_test import MockSyftObject

REPEATS = 20

PERMISSIONS = [
    ActionObjectOWNER,
    ActionObjectREAD,
    ActionObjectWRITE,
    ActionObjectEXECUTE,
]


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
def test_mongo_store_partition_sanity(
    mongo_store_partition: MongoStorePartition,
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()

    assert hasattr(mongo_store_partition, "_collection")
    assert hasattr(mongo_store_partition, "_permissions")


def test_mongo_store_partition_init_failed(root_verify_key) -> None:
    # won't connect
    mongo_config = MongoStoreClientConfig(connectTimeoutMS=1, timeoutMS=1)

    store_config = MongoStoreConfig(client_config=mongo_config)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    store = MongoStorePartition(
        root_verify_key, settings=settings, store_config=store_config
    )

    res = store.init_store()
    assert res.is_err()


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.xfail
def test_mongo_store_partition_set(
    root_verify_key, mongo_store_partition: MongoStorePartition
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()

    obj = MockSyftObject(data=1)

    res = mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)

    assert res.is_ok()
    assert res.ok() == obj
    assert (
        len(
            mongo_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 1
    )

    res = mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
    assert res.is_err()
    assert (
        len(
            mongo_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 1
    )

    res = mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=True)
    assert res.is_ok()
    assert (
        len(
            mongo_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 1
    )

    obj2 = MockSyftObject(data=2)
    res = mongo_store_partition.set(root_verify_key, obj2, ignore_duplicates=False)
    assert res.is_ok()
    assert res.ok() == obj2
    assert (
        len(
            mongo_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 2
    )

    for idx in range(REPEATS):
        obj = MockSyftObject(data=idx)
        res = mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
        assert res.is_ok()
        assert (
            len(
                mongo_store_partition.all(
                    root_verify_key,
                ).ok()
            )
            == 3 + idx
        )


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_mongo_store_partition_delete(
    root_verify_key,
    mongo_store_partition: MongoStorePartition,
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()

    objs = []
    for v in range(REPEATS):
        obj = MockSyftObject(data=v)
        mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
        objs.append(obj)

    assert len(
        mongo_store_partition.all(
            root_verify_key,
        ).ok()
    ) == len(objs)

    # random object
    obj = MockSyftObject(data="bogus")
    key = mongo_store_partition.settings.store_key.with_obj(obj)
    res = mongo_store_partition.delete(root_verify_key, key)
    assert res.is_err()
    assert len(
        mongo_store_partition.all(
            root_verify_key,
        ).ok()
    ) == len(objs)

    # cleanup store
    for idx, v in enumerate(objs):
        key = mongo_store_partition.settings.store_key.with_obj(v)
        res = mongo_store_partition.delete(root_verify_key, key)
        assert res.is_ok()
        assert (
            len(
                mongo_store_partition.all(
                    root_verify_key,
                ).ok()
            )
            == len(objs) - idx - 1
        )

        res = mongo_store_partition.delete(root_verify_key, key)
        assert res.is_err()
        assert (
            len(
                mongo_store_partition.all(
                    root_verify_key,
                ).ok()
            )
            == len(objs) - idx - 1
        )

    assert (
        len(
            mongo_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 0
    )


@pytest.mark.flaky(reruns=5, reruns_delay=2)
@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
def test_mongo_store_partition_update(
    root_verify_key,
    mongo_store_partition: MongoStorePartition,
) -> None:
    mongo_store_partition.init_store()

    # add item
    obj = MockSyftObject(data=1)
    mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
    assert (
        len(
            mongo_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 1
    )

    # fail to update missing keys
    rand_obj = MockSyftObject(data="bogus")
    key = mongo_store_partition.settings.store_key.with_obj(rand_obj)
    res = mongo_store_partition.update(root_verify_key, key, obj)
    assert res.is_err()

    # update the key multiple times
    for v in range(REPEATS):
        key = mongo_store_partition.settings.store_key.with_obj(obj)
        obj_new = MockSyftObject(data=v)

        res = mongo_store_partition.update(root_verify_key, key, obj_new)
        assert res.is_ok()

        # The ID should stay the same on update, only the values are updated.
        assert (
            len(
                mongo_store_partition.all(
                    root_verify_key,
                ).ok()
            )
            == 1
        )
        assert (
            mongo_store_partition.all(
                root_verify_key,
            )
            .ok()[0]
            .id
            == obj.id
        )
        assert (
            mongo_store_partition.all(
                root_verify_key,
            )
            .ok()[0]
            .id
            != obj_new.id
        )
        assert (
            mongo_store_partition.all(
                root_verify_key,
            )
            .ok()[0]
            .data
            == v
        )

        stored = mongo_store_partition.get_all_from_store(
            root_verify_key, QueryKeys(qks=[key])
        )
        assert stored.ok()[0].data == v


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
@pytest.mark.xfail
def test_mongo_store_partition_set_threading(
    root_verify_key,
    mongo_server_mock: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    execution_err = None
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err

        mongo_store_partition = mongo_store_partition_fn(
            root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
        )
        for idx in range(repeats):
            obj = MockObjectType(data=idx)

            for _ in range(10):
                res = mongo_store_partition.set(
                    root_verify_key, obj, ignore_duplicates=False
                )
                if res.is_ok():
                    break

            if res.is_err():
                execution_err = res
            assert res.is_ok(), res

        return execution_err

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    assert execution_err is None

    mongo_store_partition = mongo_store_partition_fn(
        root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
    )
    stored_cnt = len(
        mongo_store_partition.all(
            root_verify_key,
        ).ok()
    )
    assert stored_cnt == thread_cnt * repeats


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_mongo_store_partition_set_joblib(
    root_verify_key,
    mongo_server_mock,
) -> None:
    thread_cnt = 3
    repeats = REPEATS
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()

    def _kv_cbk(tid: int) -> None:
        for idx in range(repeats):
            mongo_store_partition = mongo_store_partition_fn(
                root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
            )
            obj = MockObjectType(data=idx)

            for _ in range(10):
                res = mongo_store_partition.set(
                    root_verify_key, obj, ignore_duplicates=False
                )
                if res.is_ok():
                    break

            if res.is_err():
                return res

        return None

    errs = Parallel(n_jobs=thread_cnt)(
        delayed(_kv_cbk)(idx) for idx in range(thread_cnt)
    )

    for execution_err in errs:
        assert execution_err is None

    mongo_store_partition = mongo_store_partition_fn(
        root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
    )
    stored_cnt = len(
        mongo_store_partition.all(
            root_verify_key,
        ).ok()
    )
    assert stored_cnt == thread_cnt * repeats


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
@pytest.mark.xfail(reason="Fails in CI sometimes")
def test_mongo_store_partition_update_threading(
    root_verify_key,
    mongo_server_mock,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()
    mongo_store_partition = mongo_store_partition_fn(
        root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
    )

    obj = MockSyftObject(data=0)
    key = mongo_store_partition.settings.store_key.with_obj(obj)
    mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err

        mongo_store_partition_local = mongo_store_partition_fn(
            root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
        )
        for repeat in range(repeats):
            obj = MockSyftObject(data=repeat)

            for _ in range(10):
                res = mongo_store_partition_local.update(root_verify_key, key, obj)
                if res.is_ok():
                    break

            if res.is_err():
                execution_err = res
            assert res.is_ok(), res

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    assert execution_err is None


@pytest.mark.xfail(reason="SyftObjectRegistry does only in-memory caching")
@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_mongo_store_partition_update_joblib(
    root_verify_key,
    mongo_server_mock: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()

    mongo_store_partition = mongo_store_partition_fn(
        root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
    )
    obj = MockSyftObject(data=0)
    key = mongo_store_partition.settings.store_key.with_obj(obj)
    mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)

    def _kv_cbk(tid: int) -> None:
        mongo_store_partition_local = mongo_store_partition_fn(
            root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
        )
        for repeat in range(repeats):
            obj = MockSyftObject(data=repeat)

            for _ in range(10):
                res = mongo_store_partition_local.update(root_verify_key, key, obj)
                if res.is_ok():
                    break

            if res.is_err():
                return res
        return None

    errs = Parallel(n_jobs=thread_cnt)(
        delayed(_kv_cbk)(idx) for idx in range(thread_cnt)
    )

    for execution_err in errs:
        assert execution_err is None


@pytest.mark.skip(reason="The tests are highly flaky, delaying progress on PR's")
@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
def test_mongo_store_partition_set_delete_threading(
    root_verify_key,
    mongo_server_mock,
) -> None:
    thread_cnt = 3
    repeats = REPEATS
    execution_err = None
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        mongo_store_partition = mongo_store_partition_fn(
            root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
        )

        for idx in range(repeats):
            obj = MockSyftObject(data=idx)

            for _ in range(10):
                res = mongo_store_partition.set(
                    root_verify_key, obj, ignore_duplicates=False
                )
                if res.is_ok():
                    break

            if res.is_err():
                execution_err = res
            assert res.is_ok()

            key = mongo_store_partition.settings.store_key.with_obj(obj)

            res = mongo_store_partition.delete(root_verify_key, key)
            if res.is_err():
                execution_err = res
            assert res.is_ok(), res

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    assert execution_err is None

    mongo_store_partition = mongo_store_partition_fn(
        root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
    )
    stored_cnt = len(
        mongo_store_partition.all(
            root_verify_key,
        ).ok()
    )
    assert stored_cnt == 0


@pytest.mark.skip(reason="The tests are highly flaky, delaying progress on PR's")
@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
def test_mongo_store_partition_set_delete_joblib(
    root_verify_key,
    mongo_server_mock,
) -> None:
    thread_cnt = 3
    repeats = REPEATS
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()

    def _kv_cbk(tid: int) -> None:
        mongo_store_partition = mongo_store_partition_fn(
            root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
        )

        for idx in range(repeats):
            obj = MockSyftObject(data=idx)

            for _ in range(10):
                res = mongo_store_partition.set(
                    root_verify_key, obj, ignore_duplicates=False
                )
                if res.is_ok():
                    break

            if res.is_err():
                return res

            key = mongo_store_partition.settings.store_key.with_obj(obj)

            res = mongo_store_partition.delete(root_verify_key, key)
            if res.is_err():
                return res
        return None

    errs = Parallel(n_jobs=thread_cnt)(
        delayed(_kv_cbk)(idx) for idx in range(thread_cnt)
    )
    for execution_err in errs:
        assert execution_err is None

    mongo_store_partition = mongo_store_partition_fn(
        root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
    )
    stored_cnt = len(
        mongo_store_partition.all(
            root_verify_key,
        ).ok()
    )
    assert stored_cnt == 0


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
def test_mongo_store_partition_permissions_collection(
    mongo_store_partition: MongoStorePartition,
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()

    collection_permissions_status = mongo_store_partition.permissions
    assert not collection_permissions_status.is_err()
    collection_permissions = collection_permissions_status.ok()
    assert isinstance(collection_permissions, MongoCollection)


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
def test_mongo_store_partition_add_remove_permission(
    root_verify_key: SyftVerifyKey, mongo_store_partition: MongoStorePartition
) -> None:
    """
    Test the add_permission and remove_permission functions of MongoStorePartition
    """
    # setting up
    res = mongo_store_partition.init_store()
    assert res.is_ok()
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
    remove_res = mongo_store_partition.remove_permission(
        ActionObjectPermission(
            uid=obj.id, permission=ActionPermission.OWNER, credentials=root_verify_key
        )
    )
    assert isinstance(remove_res, Err)
    find_res_5 = permissions_collection.find_one({"_id": obj.id})
    assert len(find_res_5["permissions"]) == 1
    assert find_res_1["permissions"] == {
        obj_read_permission.permission_string,
    }

    # there is only one permission object
    assert permissions_collection.count_documents({}) == 1

    # add permissions in a loop
    new_permissions = []
    for idx in range(1, REPEATS + 1):
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


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
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
    permissions: List[ActionObjectPermission] = [
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


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.parametrize("permission", PERMISSIONS)
def test_mongo_store_partition_has_permission(
    root_verify_key: SyftVerifyKey,
    guest_verify_key: SyftVerifyKey,
    mongo_store_partition: MongoStorePartition,
    permission: ActionObjectPermission,
) -> None:
    hacker_verify_key = SyftVerifyKey.from_string(test_verify_key_string_hacker)

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


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.parametrize("permission", PERMISSIONS)
def test_mongo_store_partition_take_ownership(
    root_verify_key: SyftVerifyKey,
    guest_verify_key: SyftVerifyKey,
    mongo_store_partition: MongoStorePartition,
    permission: ActionObjectPermission,
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()

    hacker_verify_key = SyftVerifyKey.from_string(test_verify_key_string_hacker)
    obj = MockSyftObject(data=1)

    # the guest client takes ownership of obj
    mongo_store_partition.take_ownership(uid=obj.id, credentials=guest_verify_key)
    assert mongo_store_partition.has_permission(
        permission(uid=obj.id, credentials=guest_verify_key)
    )
    # the root client will also has the permission
    assert mongo_store_partition.has_permission(
        permission(uid=obj.id, credentials=root_verify_key)
    )
    assert not mongo_store_partition.has_permission(
        permission(uid=obj.id, credentials=hacker_verify_key)
    )

    # hacker or root try to take ownership of the obj and will fail
    res = mongo_store_partition.take_ownership(
        uid=obj.id, credentials=hacker_verify_key
    )
    res_2 = mongo_store_partition.take_ownership(
        uid=obj.id, credentials=root_verify_key
    )
    assert res.is_err()
    assert res_2.is_err()
    assert res.value == res_2.value == f"UID: {obj.id} already owned."

    # another object
    obj_2 = MockSyftObject(data=2)
    # root client takes ownership
    mongo_store_partition.take_ownership(uid=obj_2.id, credentials=root_verify_key)
    assert mongo_store_partition.has_permission(
        permission(uid=obj_2.id, credentials=root_verify_key)
    )
    assert not mongo_store_partition.has_permission(
        permission(uid=obj_2.id, credentials=guest_verify_key)
    )
    assert not mongo_store_partition.has_permission(
        permission(uid=obj_2.id, credentials=hacker_verify_key)
    )


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
def test_mongo_store_partition_permissions_set(
    root_verify_key: SyftVerifyKey,
    guest_verify_key: SyftVerifyKey,
    mongo_store_partition: MongoStorePartition,
) -> None:
    """
    Test the permissions functionalities when using MongoStorePartition._set function
    """
    hacker_verify_key = SyftVerifyKey.from_string(test_verify_key_string_hacker)
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
    assert isinstance(permissions["permissions"], Set)
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


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
def test_mongo_store_partition_permissions_get_all(
    root_verify_key: SyftVerifyKey,
    guest_verify_key: SyftVerifyKey,
    mongo_store_partition: MongoStorePartition,
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()
    hacker_verify_key = SyftVerifyKey.from_string(test_verify_key_string_hacker)
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


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
def test_mongo_store_partition_permissions_delete(
    root_verify_key: SyftVerifyKey,
    guest_verify_key: SyftVerifyKey,
    mongo_store_partition: MongoStorePartition,
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()
    collection: MongoCollection = mongo_store_partition.collection.ok()
    pemissions_collection: MongoCollection = mongo_store_partition.permissions.ok()
    hacker_verify_key = SyftVerifyKey.from_string(test_verify_key_string_hacker)

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


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
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

    for v in range(REPEATS):
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
