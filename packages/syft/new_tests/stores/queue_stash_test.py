# stdlib
from pathlib import Path
from threading import Lock
from threading import Thread
from typing import Any
from typing import Generator

# third party
from pymongo import MongoClient
import pytest
from pytest_mock_resources import create_mongo_fixture

# syft absolute
from syft.core.node.new.dict_document_store import DictDocumentStore
from syft.core.node.new.dict_document_store import DictStoreConfig
from syft.core.node.new.document_store import PartitionSettings
from syft.core.node.new.mongo_client import MongoStoreClientConfig
from syft.core.node.new.mongo_document_store import MongoDocumentStore
from syft.core.node.new.mongo_document_store import MongoStoreConfig
from syft.core.node.new.queue_stash import QueueStash
from syft.core.node.new.sqlite_document_store import SQLiteDocumentStore
from syft.core.node.new.sqlite_document_store import SQLiteStoreClientConfig
from syft.core.node.new.sqlite_document_store import SQLiteStoreConfig

# relative
from .store_mocks import MockObjectType
from .store_mocks import MockSyftObject

mongo = create_mongo_fixture(scope="session")
workspace = Path("workspace")
db_name = "testing"


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
def dict_queue():
    store_config = DictStoreConfig()
    store = DictDocumentStore(store_config=store_config)
    return QueueStash(store=store)


@pytest.fixture
def sqlite_queue():
    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    store = SQLiteDocumentStore(store_config=store_config)
    return QueueStash(store=store)


@pytest.fixture
def mongo_queue(mongo):
    mongo_client = MongoClient(**mongo.pmr_credentials.as_mongo_kwargs())

    mongo_config = MongoStoreClientConfig(client=mongo_client)
    store_config = MongoStoreConfig(client_config=mongo_config, db_name=db_name)
    PartitionSettings(name="test", object_type=MockObjectType)

    mongo_client.drop_database(db_name)

    store = MongoDocumentStore(store_config=store_config)
    return QueueStash(store=store)


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue"),
        pytest.lazy_fixture("sqlite_queue"),
        pytest.lazy_fixture("mongo_queue"),
    ],
)
def test_queue_stash_sanity(queue: Any) -> None:
    assert len(queue) == 0
    assert hasattr(queue, "store")
    assert hasattr(queue, "partition")


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue"),
        pytest.lazy_fixture("sqlite_queue"),
        pytest.lazy_fixture("mongo_queue"),
    ],
)
def test_queue_stash_set_get(queue: Any) -> None:
    objs = []
    for idx in range(100):
        obj = MockSyftObject(data=idx)
        objs.append(obj)

        res = queue.set(obj, ignore_duplicates=False)
        assert res.is_ok()
        assert len(queue) == idx + 1

        res = queue.set(obj, ignore_duplicates=False)
        assert res.is_err()
        assert len(queue) == idx + 1

        assert len(queue.get_all().ok()) == idx + 1

        item = queue.find_one(id=obj.id)
        assert item.is_ok()
        assert item.ok() == obj

    cnt = len(objs)
    for obj in objs:
        res = queue.find_and_delete(id=obj.id)
        assert res.is_ok()

        cnt -= 1
        assert len(queue) == cnt
        item = queue.find_one(id=obj.id)
        assert item.is_ok()
        assert item.ok() is None


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue"),
        pytest.lazy_fixture("sqlite_queue"),
        pytest.lazy_fixture("mongo_queue"),
    ],
)
def test_queue_stash_update(queue: Any) -> None:
    obj = MockSyftObject(data=0)
    res = queue.set(obj, ignore_duplicates=False)
    assert res.is_ok()

    for idx in range(100):
        obj.data = idx

        res = queue.update(obj)
        assert res.is_ok()
        assert len(queue) == 1

        item = queue.find_one(id=obj.id)
        assert item.is_ok()
        assert item.ok().data == idx

    res = queue.find_and_delete(id=obj.id)
    assert res.is_ok()
    assert len(queue) == 0


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue"),
        pytest.lazy_fixture("sqlite_queue"),
        pytest.lazy_fixture("mongo_queue"),
    ],
)
def test_queue_set_multithreaded(queue: Any) -> None:
    thread_cnt = 3
    repeats = 100

    execution_ok = True
    lock = Lock()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_ok
        for idx in range(repeats):
            obj = MockSyftObject(data=idx)
            res = queue.set(obj, ignore_duplicates=False)

            with lock:
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
    assert len(queue) == thread_cnt * repeats


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue"),
        pytest.lazy_fixture("sqlite_queue"),
        pytest.lazy_fixture("mongo_queue"),
    ],
)
def test_queue_update_multithreaded(queue: Any) -> None:
    thread_cnt = 3
    repeats = 100

    obj = MockSyftObject(data=0)
    queue.set(obj, ignore_duplicates=False)
    execution_ok = True
    lock = Lock()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_ok
        for repeat in range(repeats):
            with lock:
                obj.data += 1
            res = queue.update(obj)

            with lock:
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
    stored = queue.find_one(id=obj.id)
    assert stored.ok().data == thread_cnt * repeats


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue"),
        pytest.lazy_fixture("sqlite_queue"),
        pytest.lazy_fixture("mongo_queue"),
    ],
)
def test_queue_set_delete_multithreaded(
    queue: Any,
) -> None:
    thread_cnt = 3
    repeats = 100

    execution_ok = True
    objs = []

    for idx in range(repeats * thread_cnt):
        obj = MockSyftObject(data=idx)
        res = queue.set(obj, ignore_duplicates=False)
        objs.append(obj)

        assert res.is_ok()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_ok
        for idx in range(repeats):
            item_idx = tid * repeats + idx

            res = queue.find_and_delete(id=objs[item_idx].id)
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
    assert len(queue) == 0
