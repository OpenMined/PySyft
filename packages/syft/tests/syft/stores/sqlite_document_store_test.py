# stdlib
import shutil
from threading import Thread
from typing import Tuple

# third party
from joblib import Parallel
from joblib import delayed

# syft absolute
from syft.core.node.new.document_store import PartitionSettings
from syft.core.node.new.document_store import QueryKeys
from syft.core.node.new.sqlite_document_store import SQLiteStoreClientConfig
from syft.core.node.new.sqlite_document_store import SQLiteStoreConfig
from syft.core.node.new.sqlite_document_store import SQLiteStorePartition

# relative
from .store_constants_test import generate_db_name
from .store_constants_test import workspace
from .store_fixtures_test import sqlite_store_partition_fn
from .store_mocks_test import MockObjectType
from .store_mocks_test import MockSyftObject

REPEATS = 20


def test_sqlite_store_partition_sanity(
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    assert hasattr(sqlite_store_partition, "data")
    assert hasattr(sqlite_store_partition, "unique_keys")
    assert hasattr(sqlite_store_partition, "searchable_keys")


def test_sqlite_store_partition_init_failed(
    sqlite_workspace: Tuple,
) -> None:
    workspace, db_name = sqlite_workspace

    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    store = SQLiteStorePartition(settings=settings, store_config=store_config)

    # delete the destination folder
    shutil.rmtree(workspace)

    res = store.init_store()
    assert res.is_err()


def test_sqlite_store_partition_set(
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    obj = MockSyftObject(data=1)
    res = sqlite_store_partition.set(obj, ignore_duplicates=False)

    assert res.is_ok()
    assert res.ok() == obj
    assert len(sqlite_store_partition.all().ok()) == 1

    res = sqlite_store_partition.set(obj, ignore_duplicates=False)
    assert res.is_err()
    assert len(sqlite_store_partition.all().ok()) == 1

    res = sqlite_store_partition.set(obj, ignore_duplicates=True)
    assert res.is_ok()
    assert len(sqlite_store_partition.all().ok()) == 1

    obj2 = MockSyftObject(data=2)
    res = sqlite_store_partition.set(obj2, ignore_duplicates=False)
    assert res.is_ok()
    assert res.ok() == obj2
    assert len(sqlite_store_partition.all().ok()) == 2

    for idx in range(REPEATS):
        obj = MockSyftObject(data=idx)
        res = sqlite_store_partition.set(obj, ignore_duplicates=False)
        assert res.is_ok()
        assert len(sqlite_store_partition.all().ok()) == 3 + idx


def test_sqlite_store_partition_set_backend_fail(
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    sqlite_db_name = generate_db_name()
    sqlite_config = SQLiteStoreClientConfig(filename=sqlite_db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    store = SQLiteStorePartition(settings=settings, store_config=store_config)
    res = store.init_store()

    # delete the db
    shutil.rmtree(workspace)

    # this should fail
    obj = MockSyftObject(data=1)

    res = sqlite_store_partition.set(obj, ignore_duplicates=False)
    assert res.is_err()


def test_sqlite_store_partition_delete(
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    objs = []
    for v in range(REPEATS):
        obj = MockSyftObject(data=v)
        sqlite_store_partition.set(obj, ignore_duplicates=False)
        objs.append(obj)

    assert len(sqlite_store_partition.all().ok()) == len(objs)

    # random object
    obj = MockSyftObject(data="bogus")
    key = sqlite_store_partition.settings.store_key.with_obj(obj)
    res = sqlite_store_partition.delete(key)
    assert res.is_err()
    assert len(sqlite_store_partition.all().ok()) == len(objs)

    # cleanup store
    for idx, v in enumerate(objs):
        key = sqlite_store_partition.settings.store_key.with_obj(v)
        res = sqlite_store_partition.delete(key)
        assert res.is_ok()
        assert len(sqlite_store_partition.all().ok()) == len(objs) - idx - 1

        res = sqlite_store_partition.delete(key)
        assert res.is_err()
        assert len(sqlite_store_partition.all().ok()) == len(objs) - idx - 1

    assert len(sqlite_store_partition.all().ok()) == 0


def test_sqlite_store_partition_update(
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    # add item
    obj = MockSyftObject(data=1)
    sqlite_store_partition.set(obj, ignore_duplicates=False)
    assert len(sqlite_store_partition.all().ok()) == 1

    # fail to update missing keys
    rand_obj = MockSyftObject(data="bogus")
    key = sqlite_store_partition.settings.store_key.with_obj(rand_obj)
    res = sqlite_store_partition.update(key, obj)
    assert res.is_err()

    # update the key multiple times
    for v in range(REPEATS):
        key = sqlite_store_partition.settings.store_key.with_obj(obj)
        obj_new = MockSyftObject(data=v)

        res = sqlite_store_partition.update(key, obj_new)
        assert res.is_ok()

        # The ID should stay the same on update, unly the values are updated.
        assert len(sqlite_store_partition.all().ok()) == 1
        assert sqlite_store_partition.all().ok()[0].id == obj.id
        assert sqlite_store_partition.all().ok()[0].id != obj_new.id
        assert sqlite_store_partition.all().ok()[0].data == v

        stored = sqlite_store_partition.get_all_from_store(QueryKeys(qks=[key]))
        assert stored.ok()[0].data == v


def test_sqlite_store_partition_set_threading(
    sqlite_workspace: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err

        sqlite_store_partition = sqlite_store_partition_fn(sqlite_workspace)
        for idx in range(repeats):
            obj = MockObjectType(data=idx)
            res = sqlite_store_partition.set(obj, ignore_duplicates=False)

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

    sqlite_store_partition = sqlite_store_partition_fn(sqlite_workspace)
    stored_cnt = len(sqlite_store_partition.all().ok())
    assert stored_cnt == thread_cnt * repeats


def test_sqlite_store_partition_set_joblib(
    sqlite_workspace: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    def _kv_cbk(tid: int) -> None:
        for idx in range(repeats):
            sqlite_store_partition = sqlite_store_partition_fn(sqlite_workspace)
            obj = MockObjectType(data=idx)
            res = sqlite_store_partition.set(obj, ignore_duplicates=False)

            if res.is_err():
                return res

        return None

    errs = Parallel(n_jobs=thread_cnt)(
        delayed(_kv_cbk)(idx) for idx in range(thread_cnt)
    )

    for execution_err in errs:
        assert execution_err is None

    sqlite_store_partition = sqlite_store_partition_fn(sqlite_workspace)
    stored_cnt = len(sqlite_store_partition.all().ok())
    assert stored_cnt == thread_cnt * repeats


def test_sqlite_store_partition_update_threading(
    sqlite_workspace: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    sqlite_store_partition = sqlite_store_partition_fn(sqlite_workspace)
    obj = MockSyftObject(data=0)
    key = sqlite_store_partition.settings.store_key.with_obj(obj)
    sqlite_store_partition.set(obj, ignore_duplicates=False)
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err

        sqlite_store_partition_local = sqlite_store_partition_fn(sqlite_workspace)
        for repeat in range(repeats):
            obj = MockSyftObject(data=repeat)
            res = sqlite_store_partition_local.update(key, obj)

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


def test_sqlite_store_partition_update_joblib(
    sqlite_workspace: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    sqlite_store_partition = sqlite_store_partition_fn(sqlite_workspace)
    obj = MockSyftObject(data=0)
    key = sqlite_store_partition.settings.store_key.with_obj(obj)
    sqlite_store_partition.set(obj, ignore_duplicates=False)

    def _kv_cbk(tid: int) -> None:
        sqlite_store_partition_local = sqlite_store_partition_fn(sqlite_workspace)
        for repeat in range(repeats):
            obj = MockSyftObject(data=repeat)
            res = sqlite_store_partition_local.update(key, obj)

            if res.is_err():
                return res
        return None

    errs = Parallel(n_jobs=thread_cnt)(
        delayed(_kv_cbk)(idx) for idx in range(thread_cnt)
    )

    for execution_err in errs:
        assert execution_err is None


def test_sqlite_store_partition_set_delete_threading(
    sqlite_workspace: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        sqlite_store_partition = sqlite_store_partition_fn(sqlite_workspace)

        for idx in range(repeats):
            obj = MockSyftObject(data=idx)
            res = sqlite_store_partition.set(obj, ignore_duplicates=False)

            if res.is_err():
                execution_err = res
            assert res.is_ok()

            key = sqlite_store_partition.settings.store_key.with_obj(obj)

            res = sqlite_store_partition.delete(key)
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

    sqlite_store_partition = sqlite_store_partition_fn(sqlite_workspace)
    stored_cnt = len(sqlite_store_partition.all().ok())
    assert stored_cnt == 0


def test_sqlite_store_partition_set_delete_joblib(
    sqlite_workspace: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    def _kv_cbk(tid: int) -> None:
        sqlite_store_partition = sqlite_store_partition_fn(sqlite_workspace)

        for idx in range(repeats):
            obj = MockSyftObject(data=idx)
            res = sqlite_store_partition.set(obj, ignore_duplicates=False)

            if res.is_err():
                return res

            key = sqlite_store_partition.settings.store_key.with_obj(obj)

            res = sqlite_store_partition.delete(key)
            if res.is_err():
                return res
        return None

    errs = Parallel(n_jobs=thread_cnt)(
        delayed(_kv_cbk)(idx) for idx in range(thread_cnt)
    )
    for execution_err in errs:
        assert execution_err is None

    sqlite_store_partition = sqlite_store_partition_fn(sqlite_workspace)
    stored_cnt = len(sqlite_store_partition.all().ok())
    assert stored_cnt == 0
