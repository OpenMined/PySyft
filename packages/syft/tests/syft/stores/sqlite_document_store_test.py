# stdlib
from pathlib import Path
import shutil
from threading import Thread
from typing import Tuple

# syft absolute
from syft.core.node.new.document_store import PartitionSettings
from syft.core.node.new.document_store import QueryKeys
from syft.core.node.new.sqlite_document_store import SQLiteStoreClientConfig
from syft.core.node.new.sqlite_document_store import SQLiteStoreConfig
from syft.core.node.new.sqlite_document_store import SQLiteStorePartition

# relative
from .store_mocks_test import MockObjectType
from .store_mocks_test import MockSyftObject


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


def test_sqlite_store_partition_set_backend_fail(
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    workspace = Path("workspace")
    db_name = "testing.sqlite"

    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    store = SQLiteStorePartition(settings=settings, store_config=store_config)
    res = store.init_store()

    # delete the db
    (workspace / db_name).unlink()

    # this should fail
    obj = MockSyftObject(data=1)

    res = sqlite_store_partition.set(obj, ignore_duplicates=False)
    assert res.is_err()


def test_sqlite_store_partition_delete(
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    objs = []
    for v in range(10):
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
    for v in range(10):
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


def test_sqlite_store_partition_set_multithreaded(
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    thread_cnt = 3
    repeats = 200

    execution_ok = True

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_ok
        for idx in range(repeats):
            obj = MockObjectType(data=idx)
            res = sqlite_store_partition.set(obj, ignore_duplicates=False)

            execution_ok &= res.is_ok()
            assert res.is_ok()

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    assert execution_ok
    stored_cnt = len(sqlite_store_partition.all().ok())
    assert stored_cnt == thread_cnt * repeats


def test_sqlite_store_partition_update_multithreaded(
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    thread_cnt = 3
    repeats = 100

    obj = MockSyftObject(data=0)
    key = sqlite_store_partition.settings.store_key.with_obj(obj)
    sqlite_store_partition.set(obj, ignore_duplicates=False)
    execution_ok = True

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_ok
        for repeat in range(repeats):
            stored = sqlite_store_partition.get_all_from_store(QueryKeys(qks=[key]))
            obj = MockSyftObject(data=stored.ok()[0].data + 1)
            res = sqlite_store_partition.update(key, obj)

            execution_ok &= res.is_ok()
            assert res.is_ok()

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    assert execution_ok
    stored = sqlite_store_partition.get_all_from_store(QueryKeys(qks=[key]))
    assert stored.ok()[0].data == thread_cnt * repeats


def test_sqlite_store_partition_set_delete_multithreaded(
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    thread_cnt = 3
    execution_ok = True

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_ok
        for idx in range(100):
            obj = MockSyftObject(data=idx)
            res = sqlite_store_partition.set(obj, ignore_duplicates=False)

            execution_ok &= res.is_ok()
            assert res.is_ok()

            key = sqlite_store_partition.settings.store_key.with_obj(obj)

            res = sqlite_store_partition.delete(key)
            execution_ok &= res.is_ok()
            assert res.is_ok()

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    assert execution_ok
    stored_cnt = len(sqlite_store_partition.all().ok())
    assert stored_cnt == 0
