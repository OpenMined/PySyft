# stdlib
import threading
from threading import Thread
import time
from typing import Any

# third party
import pytest

# syft absolute
from syft.service.queue.queue_stash import QueueItem
from syft.service.worker.worker_pool import WorkerPool
from syft.service.worker.worker_pool_service import SyftWorkerPoolService
from syft.store.linked_obj import LinkedObject
from syft.types.errors import SyftException
from syft.types.uid import UID

# relative
from .store_fixtures_test import mongo_queue_stash_fn
from .store_fixtures_test import sqlite_queue_stash_fn


def mock_queue_object():
    worker_pool_obj = WorkerPool(
        name="mypool",
        image_id=UID(),
        max_count=0,
        worker_list=[],
    )
    linked_worker_pool = LinkedObject.from_obj(
        worker_pool_obj,
        server_uid=UID(),
        service_type=SyftWorkerPoolService,
    )
    obj = QueueItem(
        id=UID(),
        server_uid=UID(),
        method="dummy_method",
        service="dummy_service",
        args=[],
        kwargs={},
        worker_pool=linked_worker_pool,
    )
    return obj


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
# @pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_queue_stash_set_get(root_verify_key, queue: Any) -> None:
    objs = []
    repeats = 5
    for idx in range(repeats):
        obj = mock_queue_object()
        objs.append(obj)

        res = queue.set(root_verify_key, obj, ignore_duplicates=False)
        assert res.is_ok()
        assert len(queue) == idx + 1

        with pytest.raises(SyftException):
            res = queue.set(root_verify_key, obj, ignore_duplicates=False)
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
        assert item.is_err()


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("dict_queue_stash"),
        pytest.lazy_fixture("sqlite_queue_stash"),
        pytest.lazy_fixture("mongo_queue_stash"),
    ],
)
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_queue_stash_update(root_verify_key, queue: Any) -> None:
    obj = mock_queue_object()
    res = queue.set(root_verify_key, obj, ignore_duplicates=False)
    assert res.is_ok()
    repeats = 5

    for idx in range(repeats):
        obj.args = [idx]

        res = queue.update(root_verify_key, obj)
        assert res.is_ok()
        assert len(queue) == 1

        item = queue.find_one(root_verify_key, id=obj.id)
        assert item.is_ok()
        assert item.ok().args == [idx]

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
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_queue_set_existing_queue_threading(root_verify_key, queue: Any) -> None:
    thread_cnt = 3
    repeats = 5

    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for _ in range(repeats):
            obj = mock_queue_object()

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
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_queue_update_existing_queue_threading(root_verify_key, queue: Any) -> None:
    thread_cnt = 3
    repeats = 5

    obj = mock_queue_object()
    queue.set(root_verify_key, obj, ignore_duplicates=False)
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        for repeat in range(repeats):
            obj.args = [repeat]

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
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_queue_set_delete_existing_queue_threading(
    root_verify_key,
    queue: Any,
) -> None:
    thread_cnt = 3
    repeats = 5

    execution_err = None
    objs = []

    for _ in range(repeats * thread_cnt):
        obj = mock_queue_object()
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
    repeats = 5

    execution_err = None
    lock = threading.Lock()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        with lock:
            queue = create_queue_cbk()

        for _ in range(repeats):
            obj = mock_queue_object()

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


@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_queue_set_sqlite(root_verify_key, sqlite_workspace):
    def create_queue_cbk():
        return sqlite_queue_stash_fn(root_verify_key, sqlite_workspace)

    helper_queue_set_threading(root_verify_key, create_queue_cbk)


@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_queue_set_threading_mongo(root_verify_key, mongo_document_store):
    def create_queue_cbk():
        return mongo_queue_stash_fn(mongo_document_store)

    helper_queue_set_threading(root_verify_key, create_queue_cbk)


def helper_queue_update_threading(root_verify_key, create_queue_cbk) -> None:
    thread_cnt = 3
    repeats = 5

    queue = create_queue_cbk()
    time.sleep(1)

    obj = mock_queue_object()
    queue.set(root_verify_key, obj, ignore_duplicates=False)
    execution_err = None
    lock = threading.Lock()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        with lock:
            queue_local = create_queue_cbk()

        for repeat in range(repeats):
            obj.args = [repeat]

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


@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_queue_update_threading_sqlite(root_verify_key, sqlite_workspace):
    def create_queue_cbk():
        return sqlite_queue_stash_fn(root_verify_key, sqlite_workspace)

    helper_queue_update_threading(root_verify_key, create_queue_cbk)


@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_queue_update_threading_mongo(root_verify_key, mongo_document_store):
    def create_queue_cbk():
        return mongo_queue_stash_fn(mongo_document_store)

    helper_queue_update_threading(root_verify_key, create_queue_cbk)


def helper_queue_set_delete_threading(
    root_verify_key,
    create_queue_cbk,
) -> None:
    thread_cnt = 3
    repeats = 5

    queue = create_queue_cbk()
    execution_err = None
    objs = []

    for _ in range(repeats * thread_cnt):
        obj = mock_queue_object()
        res = queue.set(root_verify_key, obj, ignore_duplicates=False)
        objs.append(obj)

        assert res.is_ok()

    lock = threading.Lock()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        with lock:
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


@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_queue_delete_threading_sqlite(root_verify_key, sqlite_workspace):
    def create_queue_cbk():
        return sqlite_queue_stash_fn(root_verify_key, sqlite_workspace)

    helper_queue_set_delete_threading(root_verify_key, create_queue_cbk)


@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_queue_delete_threading_mongo(root_verify_key, mongo_document_store):
    def create_queue_cbk():
        return mongo_queue_stash_fn(mongo_document_store)

    helper_queue_set_delete_threading(root_verify_key, create_queue_cbk)
