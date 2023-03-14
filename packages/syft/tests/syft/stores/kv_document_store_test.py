# stdlib
from copy import copy
from threading import Thread

# third party
import pytest

# syft absolute
from syft.core.node.new.document_store import PartitionSettings
from syft.core.node.new.document_store import QueryKeys
from syft.core.node.new.kv_document_store import KeyValueStorePartition

# relative
from .store_mocks_test import MockObjectType
from .store_mocks_test import MockStoreConfig
from .store_mocks_test import MockSyftObject


@pytest.fixture
def kv_store_partition():
    store_config = MockStoreConfig()
    settings = PartitionSettings(name="test", object_type=MockObjectType)
    store = KeyValueStorePartition(settings=settings, store_config=store_config)

    res = store.init_store()
    assert res.is_ok()

    return store


def test_kv_store_partition_sanity(kv_store_partition: KeyValueStorePartition) -> None:
    assert hasattr(kv_store_partition, "data")
    assert hasattr(kv_store_partition, "unique_keys")
    assert hasattr(kv_store_partition, "searchable_keys")


def test_kv_store_partition_init_failed() -> None:
    store_config = MockStoreConfig(is_crashed=True)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    kv_store_partition = KeyValueStorePartition(
        settings=settings, store_config=store_config
    )

    res = kv_store_partition.init_store()
    assert res.is_err()


def test_kv_store_partition_set(kv_store_partition: KeyValueStorePartition) -> None:
    obj = MockSyftObject(data=1)
    res = kv_store_partition.set(obj, ignore_duplicates=False)

    assert res.is_ok()
    assert res.ok() == obj
    assert len(kv_store_partition.all().ok()) == 1

    res = kv_store_partition.set(obj, ignore_duplicates=False)
    assert res.is_err()
    assert len(kv_store_partition.all().ok()) == 1

    res = kv_store_partition.set(obj, ignore_duplicates=True)
    assert res.is_ok()
    assert len(kv_store_partition.all().ok()) == 1

    obj2 = MockSyftObject(data=2)
    res = kv_store_partition.set(obj2, ignore_duplicates=False)
    assert res.is_ok()
    assert res.ok() == obj2
    assert len(kv_store_partition.all().ok()) == 2


def test_kv_store_partition_set_backend_fail() -> None:
    store_config = MockStoreConfig(is_crashed=True)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    kv_store_partition = KeyValueStorePartition(
        settings=settings, store_config=store_config
    )
    kv_store_partition.init_store()

    obj = MockSyftObject(data=1)

    res = kv_store_partition.set(obj, ignore_duplicates=False)
    assert res.is_err()


def test_kv_store_partition_delete(kv_store_partition: KeyValueStorePartition) -> None:
    objs = []
    for v in range(10):
        obj = MockSyftObject(data=v)
        kv_store_partition.set(obj, ignore_duplicates=False)
        objs.append(obj)

    assert len(kv_store_partition.all().ok()) == len(objs)

    # random object
    obj = MockSyftObject(data="bogus")
    key = kv_store_partition.settings.store_key.with_obj(obj)
    res = kv_store_partition.delete(key)
    assert res.is_err()
    assert len(kv_store_partition.all().ok()) == len(objs)

    # cleanup store
    for idx, v in enumerate(objs):
        key = kv_store_partition.settings.store_key.with_obj(v)
        res = kv_store_partition.delete(key)
        assert res.is_ok()
        assert len(kv_store_partition.all().ok()) == len(objs) - idx - 1

        res = kv_store_partition.delete(key)
        assert res.is_err()
        assert len(kv_store_partition.all().ok()) == len(objs) - idx - 1

    assert len(kv_store_partition.all().ok()) == 0


def test_kv_store_partition_update(kv_store_partition: KeyValueStorePartition) -> None:
    # add item
    obj = MockSyftObject(data=1)
    kv_store_partition.set(obj, ignore_duplicates=False)
    assert len(kv_store_partition.all().ok()) == 1

    # fail to update missing keys
    rand_obj = MockSyftObject(data="bogus")
    key = kv_store_partition.settings.store_key.with_obj(rand_obj)
    res = kv_store_partition.update(key, obj)
    assert res.is_err()

    # update the key multiple times
    for v in range(10):
        key = kv_store_partition.settings.store_key.with_obj(obj)
        obj_new = MockSyftObject(data=v)

        res = kv_store_partition.update(key, copy(obj_new))
        assert res.is_ok()

        # The ID should stay the same on update, unly the values are updated.
        assert len(kv_store_partition.all().ok()) == 1
        assert kv_store_partition.all().ok()[0].id == obj.id
        assert kv_store_partition.all().ok()[0].id != obj_new.id
        assert kv_store_partition.all().ok()[0].data == v

        stored = kv_store_partition.get_all_from_store(QueryKeys(qks=[key]))
        assert stored.ok()[0].data == v


def test_kv_store_partition_set_multithreaded(
    kv_store_partition: KeyValueStorePartition,
) -> None:
    thread_cnt = 5
    repeats = 50
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for idx in range(repeats):
            obj = MockSyftObject(data=idx)
            res = kv_store_partition.set(obj, ignore_duplicates=False)

            if res.is_err():
                execution_err = res
            assert res.is_ok()

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    stored = kv_store_partition.all()

    assert execution_err is None
    stored_cnt = len(stored.ok())
    assert stored_cnt == thread_cnt * repeats


def test_kv_store_partition_update_multithreaded(
    kv_store_partition: KeyValueStorePartition,
) -> None:
    thread_cnt = 5
    repeats = 50

    obj = MockSyftObject(data=0)
    key = kv_store_partition.settings.store_key.with_obj(obj)
    kv_store_partition.set(obj, ignore_duplicates=False)
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for repeat in range(repeats):
            obj = MockSyftObject(data=repeat)
            res = kv_store_partition.update(key, obj)

            if res.is_err():
                execution_err = res
            assert res.is_ok()

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    assert execution_err is None


def test_kv_store_partition_set_delete_multithreaded(
    kv_store_partition: KeyValueStorePartition,
) -> None:
    thread_cnt = 5
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for idx in range(50):
            obj = MockSyftObject(data=idx)
            res = kv_store_partition.set(obj, ignore_duplicates=False)

            if res.is_err():
                execution_err = res
            assert res.is_ok()

            key = kv_store_partition.settings.store_key.with_obj(obj)

            res = kv_store_partition.delete(key)
            if res.is_err():
                execution_err = res
            assert res.is_ok()

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    assert execution_err is None
    stored_cnt = len(kv_store_partition.all().ok())
    assert stored_cnt == 0
