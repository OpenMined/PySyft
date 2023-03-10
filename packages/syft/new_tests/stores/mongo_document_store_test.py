# stdlib
from threading import Thread

# third party
from pymongo import MongoClient
import pytest
from pytest_mock_resources import create_mongo_fixture

# syft absolute
from syft.core.node.new.document_store import PartitionSettings
from syft.core.node.new.document_store import QueryKeys
from syft.core.node.new.mongo_client import MongoStoreClientConfig
from syft.core.node.new.mongo_document_store import MongoStoreConfig
from syft.core.node.new.mongo_document_store import MongoStorePartition

# relative
from .store_mocks import MockObjectType
from .store_mocks import MockSyftObject

mongo = create_mongo_fixture(scope="session")
db_name = "testing"


@pytest.fixture
def store(mongo):
    mongo_client = MongoClient(**mongo.pmr_credentials.as_mongo_kwargs())

    mongo_config = MongoStoreClientConfig(client=mongo_client)
    store_config = MongoStoreConfig(client_config=mongo_config, db_name=db_name)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    yield MongoStorePartition(settings=settings, store_config=store_config)

    mongo_client.drop_database(db_name)


def test_mongo_store_partition_sanity(store: MongoStorePartition) -> None:
    res = store.init_store()
    assert res.is_ok()

    assert hasattr(store, "_collection")


def test_mongo_store_partition_init_failed() -> None:
    # won't connect
    mongo_config = MongoStoreClientConfig(connectTimeoutMS=1, timeoutMS=1)

    store_config = MongoStoreConfig(client_config=mongo_config)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    return MongoStorePartition(settings=settings, store_config=store_config)

    res = store.init_store()
    assert res.is_err()


def test_mongo_store_partition_set(store: MongoStorePartition) -> None:
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


def test_mongo_store_partition_delete(store: MongoStorePartition) -> None:
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


def test_mongo_store_partition_update(store: MongoStorePartition) -> None:
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


def test_mongo_store_partition_set_multithreaded(store: MongoStorePartition) -> None:
    thread_cnt = 3
    store.init_store()

    execution_ok = True

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_ok
        for idx in range(100):
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
    assert stored_cnt == 1000


def test_mongo_store_partition_update_multithreaded(
    store: MongoStorePartition,
) -> None:
    store.init_store()

    obj = MockSyftObject(data=0)
    key = store.settings.store_key.with_obj(obj)
    store.set(obj, ignore_duplicates=False)
    execution_ok = True

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_ok
        for repeat in range(100):
            stored = store.get_all_from_store(QueryKeys(qks=[key]))
            obj = MockSyftObject(data=stored.ok()[0].data + 1)
            res = store.update(key, obj)

            execution_ok &= res.is_ok()
