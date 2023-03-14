# stdlib
from threading import Lock
from threading import Thread
from typing import Any

# third party
import pytest

# relative
from .store_mocks_test import MockSyftObject


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue_stash"),
        pytest.lazy_fixture("sqlite_queue_stash"),
        pytest.lazy_fixture("mongo_queue_stash"),
    ],
)
def test_queue_stash_sanity(queue: Any) -> None:
    assert len(queue) == 0
    assert hasattr(queue, "store")
    assert hasattr(queue, "partition")


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue_stash"),
        pytest.lazy_fixture("sqlite_queue_stash"),
        pytest.lazy_fixture("mongo_queue_stash"),
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
        pytest.lazy_fixture("dict_queue_stash"),
        pytest.lazy_fixture("sqlite_queue_stash"),
        pytest.lazy_fixture("mongo_queue_stash"),
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
        pytest.lazy_fixture("dict_queue_stash"),
        pytest.lazy_fixture("sqlite_queue_stash"),
        pytest.lazy_fixture("mongo_queue_stash"),
    ],
)
def test_queue_set_multithreaded(queue: Any) -> None:
    thread_cnt = 5
    repeats = 50

    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for idx in range(repeats):
            obj = MockSyftObject(data=idx)
            res = queue.set(obj, ignore_duplicates=False)

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
    assert len(queue) == thread_cnt * repeats


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue_stash"),
        pytest.lazy_fixture("sqlite_queue_stash"),
        pytest.lazy_fixture("mongo_queue_stash"),
    ],
)
def test_queue_update_multithreaded(queue: Any) -> None:
    thread_cnt = 5
    repeats = 50

    obj = MockSyftObject(data=0)
    queue.set(obj, ignore_duplicates=False)
    execution_err = None
    Lock()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for repeat in range(repeats):
            obj.data = repeat
            res = queue.update(obj)

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


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue_stash"),
        pytest.lazy_fixture("sqlite_queue_stash"),
        pytest.lazy_fixture("mongo_queue_stash"),
    ],
)
def test_queue_set_delete_multithreaded(
    queue: Any,
) -> None:
    thread_cnt = 5
    repeats = 50

    execution_err = None
    objs = []

    for idx in range(repeats * thread_cnt):
        obj = MockSyftObject(data=idx)
        res = queue.set(obj, ignore_duplicates=False)
        objs.append(obj)

        assert res.is_ok()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for idx in range(repeats):
            item_idx = tid * repeats + idx

            res = queue.find_and_delete(id=objs[item_idx].id)
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
    assert len(queue) == 0
