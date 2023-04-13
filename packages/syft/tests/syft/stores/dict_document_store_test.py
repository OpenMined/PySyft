# stdlib
from threading import Thread

# syft absolute
from syft.core.node.new.dict_document_store import DictStorePartition
from syft.core.node.new.document_store import QueryKeys

# relative
from .store_mocks_test import MockObjectType
from .store_mocks_test import MockSyftObject


def test_dict_store_partition_sanity(dict_store_partition: DictStorePartition) -> None:
    res = dict_store_partition.init_store()
    assert res.is_ok()

    assert hasattr(dict_store_partition, "data")
    assert hasattr(dict_store_partition, "unique_keys")
    assert hasattr(dict_store_partition, "searchable_keys")


def test_dict_store_partition_set(dict_store_partition: DictStorePartition) -> None:
    res = dict_store_partition.init_store()
    assert res.is_ok()

    obj = MockSyftObject(data=1)
    res = dict_store_partition.set(obj, ignore_duplicates=False)

    assert res.is_ok()
    assert res.ok() == obj
    assert len(dict_store_partition.all().ok()) == 1

    res = dict_store_partition.set(obj, ignore_duplicates=False)
    assert res.is_err()
    assert len(dict_store_partition.all().ok()) == 1

    res = dict_store_partition.set(obj, ignore_duplicates=True)
    assert res.is_ok()
    assert len(dict_store_partition.all().ok()) == 1

    obj2 = MockSyftObject(data=2)
    res = dict_store_partition.set(obj2, ignore_duplicates=False)
    assert res.is_ok()
    assert res.ok() == obj2
    assert len(dict_store_partition.all().ok()) == 2

    for idx in range(100):
        obj = MockSyftObject(data=idx)
        res = dict_store_partition.set(obj, ignore_duplicates=False)
        assert res.is_ok()
        assert len(dict_store_partition.all().ok()) == 3 + idx


def test_dict_store_partition_delete(dict_store_partition: DictStorePartition) -> None:
    res = dict_store_partition.init_store()
    assert res.is_ok()

    objs = []
    for v in range(10):
        obj = MockSyftObject(data=v)
        dict_store_partition.set(obj, ignore_duplicates=False)
        objs.append(obj)

    assert len(dict_store_partition.all().ok()) == len(objs)

    # random object
    obj = MockSyftObject(data="bogus")
    key = dict_store_partition.settings.store_key.with_obj(obj)
    res = dict_store_partition.delete(key)
    assert res.is_err()
    assert len(dict_store_partition.all().ok()) == len(objs)

    # cleanup store
    for idx, v in enumerate(objs):
        key = dict_store_partition.settings.store_key.with_obj(v)
        res = dict_store_partition.delete(key)
        assert res.is_ok()
        assert len(dict_store_partition.all().ok()) == len(objs) - idx - 1

        res = dict_store_partition.delete(key)
        assert res.is_err()
        assert len(dict_store_partition.all().ok()) == len(objs) - idx - 1

    assert len(dict_store_partition.all().ok()) == 0


def test_dict_store_partition_update(dict_store_partition: DictStorePartition) -> None:
    dict_store_partition.init_store()

    # add item
    obj = MockSyftObject(data=1)
    dict_store_partition.set(obj, ignore_duplicates=False)
    assert len(dict_store_partition.all().ok()) == 1

    # fail to update missing keys
    rand_obj = MockSyftObject(data="bogus")
    key = dict_store_partition.settings.store_key.with_obj(rand_obj)
    res = dict_store_partition.update(key, obj)
    assert res.is_err()

    # update the key multiple times
    for v in range(10):
        key = dict_store_partition.settings.store_key.with_obj(obj)
        obj_new = MockSyftObject(data=v)

        res = dict_store_partition.update(key, obj_new)
        assert res.is_ok()

        # The ID should stay the same on update, unly the values are updated.
        assert len(dict_store_partition.all().ok()) == 1
        assert dict_store_partition.all().ok()[0].id == obj.id
        assert dict_store_partition.all().ok()[0].id != obj_new.id
        assert dict_store_partition.all().ok()[0].data == v

        stored = dict_store_partition.get_all_from_store(QueryKeys(qks=[key]))
        assert stored.ok()[0].data == v


def test_dict_store_partition_set_multithreaded(
    dict_store_partition: DictStorePartition,
) -> None:
    thread_cnt = 5
    repeats = 200

    dict_store_partition.init_store()

    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for idx in range(repeats):
            obj = MockObjectType(data=idx)
            for retry in range(10):
                res = dict_store_partition.set(obj, ignore_duplicates=False)
                if res.is_ok():
                    break

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
    stored_cnt = len(dict_store_partition.all().ok())
    assert stored_cnt == repeats * thread_cnt


def test_dict_store_partition_update_multithreaded(
    dict_store_partition: DictStorePartition,
) -> None:
    thread_cnt = 5
    repeats = 200
    dict_store_partition.init_store()

    obj = MockSyftObject(data=0)
    key = dict_store_partition.settings.store_key.with_obj(obj)
    dict_store_partition.set(obj, ignore_duplicates=False)
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for repeat in range(repeats):
            obj = MockSyftObject(data=repeat)
            for retry in range(10):
                res = dict_store_partition.update(key, obj)
                if res.is_ok():
                    break

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


def test_dict_store_partition_set_delete_multithreaded(
    dict_store_partition: DictStorePartition,
) -> None:
    dict_store_partition.init_store()

    thread_cnt = 5
    repeats = 200

    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for idx in range(repeats):
            obj = MockSyftObject(data=idx)
            for retry in range(10):
                res = dict_store_partition.set(obj, ignore_duplicates=False)
                if res.is_ok():
                    break

            if res.is_err():
                execution_err = res
            assert res.is_ok()

            key = dict_store_partition.settings.store_key.with_obj(obj)

            res = dict_store_partition.delete(key)
            if res.is_err():
                execution_err = res

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    assert execution_err is None
    stored_cnt = len(dict_store_partition.all().ok())
    assert stored_cnt == 0
