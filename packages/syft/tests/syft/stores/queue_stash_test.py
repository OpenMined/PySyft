# stdlib
import sys
from threading import Thread
from typing import Any

# third party
from joblib import Parallel
from joblib import delayed
import pytest

# relative
from .store_fixtures_test import mongo_queue_stash_fn
from .store_fixtures_test import sqlite_queue_stash_fn
from .store_mocks_test import MockSyftObject

REPEATS = 20


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue_stash"),
        pytest.lazy_fixture("sqlite_queue_stash"),
        pytest.lazy_fixture("mongo_queue_stash"),
    ],
)
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="pytest_mock_resources + docker issues on Windows and OSX",
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
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="pytest_mock_resources + docker issues on Windows and OSX",
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_queue_stash_set_get(root_verify_key, queue: Any) -> None:
    objs = []
    for idx in range(REPEATS):
        obj = MockSyftObject(data=idx)
        objs.append(obj)

        res = queue.set(root_verify_key, obj, ignore_duplicates=False)
        assert res.is_ok()
        assert len(queue) == idx + 1

        res = queue.set(root_verify_key, obj, ignore_duplicates=False)
        assert res.is_err()
        assert len(queue) == idx + 1

        assert len(queue.get_all(root_verify_key).ok()) == idx + 1

        item = queue.find_one(root_verify_key, id=obj.id)
        assert item.is_ok()
        assert item.ok() == obj

    cnt = len(objs)
    for obj in objs:
        res = queue.find_and_delete(root_verify_key, id=obj.id)
        assert res.is_ok()

        cnt -= 1
        assert len(queue) == cnt
        item = queue.find_one(root_verify_key, id=obj.id)
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
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="pytest_mock_resources + docker issues on Windows or OSX",
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_queue_stash_update(root_verify_key, queue: Any) -> None:
    obj = MockSyftObject(data=0)
    res = queue.set(root_verify_key, obj, ignore_duplicates=False)
    assert res.is_ok()

    for idx in range(REPEATS):
        obj.data = idx

        res = queue.update(root_verify_key, obj)
        assert res.is_ok()
        assert len(queue) == 1

        item = queue.find_one(root_verify_key, id=obj.id)
        assert item.is_ok()
        assert item.ok().data == idx

    res = queue.find_and_delete(root_verify_key, id=obj.id)
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
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="pytest_mock_resources + docker issues on Windows or OSX",
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
@pytest.mark.xfail
def test_queue_set_existing_queue_threading(root_verify_key, queue: Any) -> None:
    thread_cnt = 3
    repeats = REPEATS

    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for idx in range(repeats):
            obj = MockSyftObject(data=idx)

            for _ in range(10):
                res = queue.set(root_verify_key, obj, ignore_duplicates=False)
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
    assert len(queue) == thread_cnt * repeats


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue_stash"),
        pytest.lazy_fixture("sqlite_queue_stash"),
        pytest.lazy_fixture("mongo_queue_stash"),
    ],
)
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="pytest_mock_resources + docker issues on Windows or OSX",
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_queue_update_existing_queue_threading(root_verify_key, queue: Any) -> None:
    thread_cnt = 3
    repeats = REPEATS

    obj = MockSyftObject(data=0)
    queue.set(root_verify_key, obj, ignore_duplicates=False)
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for repeat in range(repeats):
            obj.data = repeat

            for _ in range(10):
                res = queue.update(root_verify_key, obj)
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


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue_stash"),
        pytest.lazy_fixture("sqlite_queue_stash"),
        pytest.lazy_fixture("mongo_queue_stash"),
    ],
)
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="pytest_mock_resources + docker issues on Windows or OSX",
)
@pytest.mark.flaky(reruns=10, reruns_delay=2)
def test_queue_set_delete_existing_queue_threading(
    root_verify_key,
    queue: Any,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    execution_err = None
    objs = []

    for idx in range(repeats * thread_cnt):
        obj = MockSyftObject(data=idx)
        res = queue.set(root_verify_key, obj, ignore_duplicates=False)
        objs.append(obj)

        assert res.is_ok()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for idx in range(repeats):
            item_idx = tid * repeats + idx

            for _ in range(10):
                res = queue.find_and_delete(root_verify_key, id=objs[item_idx].id)
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
    assert len(queue) == 0


def helper_queue_set_threading(root_verify_key, create_queue_cbk) -> None:
    thread_cnt = 3
    repeats = REPEATS

    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        queue = create_queue_cbk()

        for idx in range(repeats):
            obj = MockSyftObject(data=idx)

            for _ in range(10):
                res = queue.set(root_verify_key, obj, ignore_duplicates=False)
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

    queue = create_queue_cbk()

    assert execution_err is None
    assert len(queue) == thread_cnt * repeats


def helper_queue_set_joblib(root_verify_key, create_queue_cbk) -> None:
    thread_cnt = 3
    repeats = 10

    def _kv_cbk(tid: int) -> None:
        queue = create_queue_cbk()

        for idx in range(repeats):
            obj = MockSyftObject(data=idx)

            for _ in range(10):
                res = queue.set(root_verify_key, obj, ignore_duplicates=False)
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

    queue = create_queue_cbk()
    assert len(queue) == thread_cnt * repeats


@pytest.mark.parametrize(
    "backend", [helper_queue_set_threading, helper_queue_set_joblib]
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_queue_set_sqlite(root_verify_key, sqlite_workspace, backend):
    def create_queue_cbk():
        return sqlite_queue_stash_fn(root_verify_key, sqlite_workspace)

    backend(root_verify_key, create_queue_cbk)


@pytest.mark.xfail(
    reason="MongoDocumentStore is not serializable, but the same instance is needed for the partitions"
)
@pytest.mark.parametrize(
    "backend", [helper_queue_set_threading, helper_queue_set_joblib]
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_queue_set_threading_mongo(mongo_document_store, backend):
    def create_queue_cbk():
        return mongo_queue_stash_fn(mongo_document_store)

    backend(create_queue_cbk)


def helper_queue_update_threading(root_verify_key, create_queue_cbk) -> None:
    thread_cnt = 3
    repeats = REPEATS

    queue = create_queue_cbk()

    obj = MockSyftObject(data=0)
    queue.set(root_verify_key, obj, ignore_duplicates=False)
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        queue_local = create_queue_cbk()

        for repeat in range(repeats):
            obj.data = repeat

            for _ in range(10):
                res = queue_local.update(root_verify_key, obj)
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


def helper_queue_update_joblib(root_verify_key, create_queue_cbk) -> None:
    thread_cnt = 3
    repeats = REPEATS

    def _kv_cbk(tid: int) -> None:
        queue_local = create_queue_cbk()

        for repeat in range(repeats):
            obj.data = repeat

            for _ in range(10):
                res = queue_local.update(root_verify_key, obj)
                if res.is_ok():
                    break

            if res.is_err():
                return res
        return None

    queue = create_queue_cbk()

    obj = MockSyftObject(data=0)
    queue.set(root_verify_key, obj, ignore_duplicates=False)

    errs = Parallel(n_jobs=thread_cnt)(
        delayed(_kv_cbk)(idx) for idx in range(thread_cnt)
    )
    for execution_err in errs:
        assert execution_err is None


@pytest.mark.parametrize(
    "backend", [helper_queue_update_threading, helper_queue_update_joblib]
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_queue_update_threading_sqlite(root_verify_key, sqlite_workspace, backend):
    def create_queue_cbk():
        return sqlite_queue_stash_fn(root_verify_key, sqlite_workspace)

    backend(root_verify_key, create_queue_cbk)


@pytest.mark.xfail(
    reason="MongoDocumentStore is not serializable, but the same instance is needed for the partitions"
)
@pytest.mark.parametrize(
    "backend", [helper_queue_update_threading, helper_queue_update_joblib]
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_queue_update_threading_mongo(mongo_document_store, backend):
    def create_queue_cbk():
        return mongo_queue_stash_fn(mongo_document_store)

    backend(create_queue_cbk)


def helper_queue_set_delete_threading(
    root_verify_key,
    create_queue_cbk,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    queue = create_queue_cbk()
    execution_err = None
    objs = []

    for idx in range(repeats * thread_cnt):
        obj = MockSyftObject(data=idx)
        res = queue.set(root_verify_key, obj, ignore_duplicates=False)
        objs.append(obj)

        assert res.is_ok()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        queue = create_queue_cbk()
        for idx in range(repeats):
            item_idx = tid * repeats + idx

            for _ in range(10):
                res = queue.find_and_delete(root_verify_key, id=objs[item_idx].id)
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
    assert len(queue) == 0


def helper_queue_set_delete_joblib(
    root_verify_key,
    create_queue_cbk,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        queue = create_queue_cbk()
        for idx in range(repeats):
            item_idx = tid * repeats + idx

            for _ in range(10):
                res = queue.find_and_delete(root_verify_key, id=objs[item_idx].id)
                if res.is_ok():
                    break

            if res.is_err():
                execution_err = res
            assert res.is_ok()

    queue = create_queue_cbk()
    execution_err = None
    objs = []

    for idx in range(repeats * thread_cnt):
        obj = MockSyftObject(data=idx)
        res = queue.set(root_verify_key, obj, ignore_duplicates=False)
        objs.append(obj)

        assert res.is_ok()

    errs = Parallel(n_jobs=thread_cnt)(
        delayed(_kv_cbk)(idx) for idx in range(thread_cnt)
    )

    for execution_err in errs:
        assert execution_err is None

    assert len(queue) == 0


@pytest.mark.parametrize(
    "backend", [helper_queue_set_delete_threading, helper_queue_set_delete_joblib]
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_queue_delete_threading_sqlite(root_verify_key, sqlite_workspace, backend):
    def create_queue_cbk():
        return sqlite_queue_stash_fn(root_verify_key, sqlite_workspace)

    backend(root_verify_key, create_queue_cbk)


@pytest.mark.xfail(
    reason="MongoDocumentStore is not serializable, but the same instance is needed for the partitions"
)
@pytest.mark.parametrize(
    "backend", [helper_queue_set_delete_threading, helper_queue_set_delete_joblib]
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_queue_delete_threading_mongo(mongo_document_store, backend):
    def create_queue_cbk():
        return mongo_queue_stash_fn(mongo_document_store)

    backend(create_queue_cbk)
