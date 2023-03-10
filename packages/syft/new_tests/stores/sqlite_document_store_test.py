# stdlib
from pathlib import Path
import shutil
from threading import Thread
from typing import Generator

# third party
import pytest

# syft absolute
from syft.core.node.new.document_store import PartitionSettings
from syft.core.node.new.document_store import QueryKeys
from syft.core.node.new.sqlite_document_store import SQLiteStoreClientConfig
from syft.core.node.new.sqlite_document_store import SQLiteStoreConfig
from syft.core.node.new.sqlite_document_store import SQLiteStorePartition

# relative
from .store_mocks import MockObjectType
from .store_mocks import MockSyftObject

workspace = Path("workspace")
db_name = "testing.sqlite"


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
def store():
    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    return SQLiteStorePartition(settings=settings, store_config=store_config)


def test_sqlite_store_partition_sanity(store: SQLiteStorePartition) -> None:
    res = store.init_store()
    assert res.is_ok()

    assert hasattr(store, "data")
    assert hasattr(store, "unique_keys")
    assert hasattr(store, "searchable_keys")


def test_sqlite_store_partition_init_failed(store: SQLiteStorePartition) -> None:
    # delete the destination folder
    shutil.rmtree(workspace)

    res = store.init_store()
    assert res.is_err()


def test_sqlite_store_partition_set(store: SQLiteStorePartition) -> None:
    res = store.init_store()
    assert res.is_ok()

    obj = MockSyftObject(data=1)
    res = store.set(obj, ignore_duplicates=False)

    assert res.is_ok()
    assert res.ok() == obj
    assert len(store.all().ok()) == 1

    res = store.set(obj, ignore_duplicates=False)
    assert res.is_err()
    assert len(store.all().ok()) == 1

    res = store.set(obj, ignore_duplicates=True)
    assert res.is_ok()
    assert len(store.all().ok()) == 1

    obj2 = MockSyftObject(data=2)
    res = store.set(obj2, ignore_duplicates=False)
    assert res.is_ok()
    assert res.ok() == obj2
    assert len(store.all().ok()) == 2


def test_sqlite_store_partition_set_backend_fail(store: SQLiteStorePartition) -> None:
    res = store.init_store()
    assert res.is_ok()

    # delete the db
    (workspace / db_name).unlink()

    # this should fail
    obj = MockSyftObject(data=1)

    res = store.set(obj, ignore_duplicates=False)
    assert res.is_err()


def test_sqlite_store_partition_delete(store: SQLiteStorePartition) -> None:
    res = store.init_store()
    assert res.is_ok()

    objs = []
    for v in range(10):
        obj = MockSyftObject(data=v)
        store.set(obj, ignore_duplicates=False)
        objs.append(obj)

    assert len(store.all().ok()) == len(objs)

    # random object
    obj = MockSyftObject(data="bogus")
    key = store.settings.store_key.with_obj(obj)
    res = store.delete(key)
    assert res.is_err()
    assert len(store.all().ok()) == len(objs)

    # cleanup store
    for idx, v in enumerate(objs):
        key = store.settings.store_key.with_obj(v)
        res = store.delete(key)
        assert res.is_ok()
        assert len(store.all().ok()) == len(objs) - idx - 1

        res = store.delete(key)
        assert res.is_err()
        assert len(store.all().ok()) == len(objs) - idx - 1

    assert len(store.all().ok()) == 0


def test_sqlite_store_partition_update(store: SQLiteStorePartition) -> None:
    store.init_store()

    # add item
    obj = MockSyftObject(data=1)
    store.set(obj, ignore_duplicates=False)
    assert len(store.all().ok()) == 1

    # fail to update missing keys
    rand_obj = MockSyftObject(data="bogus")
    key = store.settings.store_key.with_obj(rand_obj)
    res = store.update(key, obj)
    assert res.is_err()

    # update the key multiple times
    for v in range(10):
        key = store.settings.store_key.with_obj(obj)
        obj_new = MockSyftObject(data=v)

        res = store.update(key, obj_new)
        assert res.is_ok()

        assert len(store.all().ok()) == 1
        assert store.all().ok()[0].id != obj.id
        assert store.all().ok()[0].id == obj_new.id
        assert store.all().ok()[0].data == v

        stored = store.get_all_from_store(QueryKeys(qks=[key]))
        assert stored.ok()[0].data == v


def test_sqlite_store_partition_set_multithreaded(store: SQLiteStorePartition) -> None:
    thread_cnt = 3
    repeats = 100

    store.init_store()

    execution_ok = True

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_ok
        for idx in range(repeats):
            obj = MockObjectType(data=idx)
            res = store.set(obj, ignore_duplicates=False)

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
    stored_cnt = len(store.all().ok())
    assert stored_cnt == thread_cnt * repeats


def test_sqlite_store_partition_update_multithreaded(
    store: SQLiteStorePartition,
) -> None:
    thread_cnt = 3
    repeats = 100

    store.init_store()

    obj = MockSyftObject(data=0)
    key = store.settings.store_key.with_obj(obj)
    store.set(obj, ignore_duplicates=False)
    execution_ok = True

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_ok
        for repeat in range(repeats):
            stored = store.get_all_from_store(QueryKeys(qks=[key]))
            obj = MockSyftObject(data=stored.ok()[0].data + 1)
            res = store.update(key, obj)

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
    stored = store.get_all_from_store(QueryKeys(qks=[key]))
    assert stored.ok()[0].data == thread_cnt * repeats


def test_sqlite_store_partition_set_delete_multithreaded(
    store: SQLiteStorePartition,
) -> None:
    store.init_store()

    thread_cnt = 3
    execution_ok = True

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_ok
        for idx in range(100):
            obj = MockSyftObject(data=idx)
            res = store.set(obj, ignore_duplicates=False)

            execution_ok &= res.is_ok()
            assert res.is_ok()

            key = store.settings.store_key.with_obj(obj)

            res = store.delete(key)
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
    stored_cnt = len(store.all().ok())
    assert stored_cnt == 0
