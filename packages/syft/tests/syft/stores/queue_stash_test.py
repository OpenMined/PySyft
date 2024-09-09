# stdlib
from concurrent.futures import ThreadPoolExecutor

# third party
import pytest

# syft absolute
from syft.service.queue.queue_stash import QueueItem
from syft.service.queue.queue_stash import QueueStash
from syft.service.worker.worker_pool import WorkerPool
from syft.service.worker.worker_pool_service import SyftWorkerPoolService
from syft.store.linked_obj import LinkedObject
from syft.types.errors import SyftException
from syft.types.uid import UID


def mock_queue_object() -> QueueItem:
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
        pytest.lazy_fixture("queue_stash"),
    ],
)
def test_queue_stash_sanity(queue: QueueStash) -> None:
    assert len(queue) == 0


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("queue_stash"),
    ],
)
#
def test_queue_stash_set_get(root_verify_key, queue: QueueStash) -> None:
    objs: list[QueueItem] = []
    repeats = 5
    for idx in range(repeats):
        obj = mock_queue_object()
        objs.append(obj)

        queue.set(root_verify_key, obj, ignore_duplicates=False).unwrap()
        assert len(queue) == idx + 1

        with pytest.raises(SyftException):
            queue.set(root_verify_key, obj, ignore_duplicates=False).unwrap()
        assert len(queue) == idx + 1

        assert len(queue.get_all(root_verify_key).ok()) == idx + 1

        item = queue.get_by_uid(root_verify_key, uid=obj.id).unwrap()
        assert item == obj

    cnt = len(objs)
    for obj in objs:
        queue.delete_by_uid(root_verify_key, uid=obj.id).unwrap()
        cnt -= 1
        assert len(queue) == cnt
        item = queue.get_by_uid(root_verify_key, uid=obj.id)
        assert item.is_err()


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("queue_stash"),
    ],
)
def test_queue_stash_update(queue: QueueStash) -> None:
    root_verify_key = queue.db.root_verify_key
    obj = mock_queue_object()
    queue.set(root_verify_key, obj, ignore_duplicates=False).unwrap()
    repeats = 5

    for idx in range(repeats):
        obj.args = [idx]

        queue.update(root_verify_key, obj).unwrap()
        assert len(queue) == 1

        item = queue.get_by_uid(root_verify_key, uid=obj.id).unwrap()
        assert item.args == [idx]

    queue.delete_by_uid(root_verify_key, uid=obj.id).unwrap()
    assert len(queue) == 0


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("queue_stash"),
    ],
)
def test_queue_set_existing_queue_threading(root_verify_key, queue: QueueStash) -> None:
    root_verify_key = queue.db.root_verify_key
    items_to_create = 100
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                lambda obj: queue.set(
                    root_verify_key,
                    mock_queue_object(),
                ),
                range(items_to_create),
            )
        )
        assert all(res.is_ok() for res in results), "Error occurred during execution"
    assert len(queue) == items_to_create


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("queue_stash"),
    ],
)
def test_queue_update_existing_queue_threading(queue: QueueStash) -> None:
    root_verify_key = queue.db.root_verify_key
    obj = mock_queue_object()

    def update_queue():
        obj.args = [UID()]
        res = queue.update(root_verify_key, obj)
        return res

    queue.set(root_verify_key, obj, ignore_duplicates=False)

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Run the update_queue function in multiple threads
        results = list(
            executor.map(
                lambda _: update_queue(),
                range(5),
            )
        )
        assert all(res.is_ok() for res in results), "Error occurred during execution"

    assert len(queue) == 1
    item = queue.get_by_uid(root_verify_key, uid=obj.id).unwrap()
    assert item.args != []


@pytest.mark.parametrize(
    "queue",
    [
        pytest.lazy_fixture("queue_stash"),
    ],
)
def test_queue_set_delete_existing_queue_threading(
    queue: QueueStash,
) -> None:
    root_verify_key = queue.db.root_verify_key
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                lambda obj: queue.set(
                    root_verify_key,
                    mock_queue_object(),
                ),
                range(15),
            )
        )
        objs = [item.unwrap() for item in results]

        results = list(
            executor.map(
                lambda obj: queue.delete_by_uid(root_verify_key, uid=obj.id),
                objs,
            )
        )
        assert all(res.is_ok() for res in results), "Error occurred during execution"


def test_queue_set(queue_stash: QueueStash):
    root_verify_key = queue_stash.db.root_verify_key
    config = queue_stash.db.config
    server_uid = queue_stash.db.server_uid

    def set_in_new_thread(_):
        queue_stash = QueueStash.random(
            root_verify_key=root_verify_key,
            config=config,
            server_uid=server_uid,
        )
        return queue_stash.set(root_verify_key, mock_queue_object())

    total_repeats = 50
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                set_in_new_thread,
                range(total_repeats),
            )
        )

    assert all(res.is_ok() for res in results), "Error occurred during execution"
    assert len(queue_stash) == total_repeats


def test_queue_update_threading(queue_stash: QueueStash):
    root_verify_key = queue_stash.db.root_verify_key
    config = queue_stash.db.config
    server_uid = queue_stash.db.server_uid
    obj = mock_queue_object()
    queue_stash.set(root_verify_key, obj).unwrap()

    def update_in_new_thread(_):
        queue_stash = QueueStash.random(
            root_verify_key=root_verify_key,
            config=config,
            server_uid=server_uid,
        )
        obj.args = [UID()]
        return queue_stash.update(root_verify_key, obj)

    total_repeats = 50
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                update_in_new_thread,
                range(total_repeats),
            )
        )

    assert all(res.is_ok() for res in results), "Error occurred during execution"
    assert len(queue_stash) == 1


def test_queue_delete_threading(queue_stash: QueueStash):
    root_verify_key = queue_stash.db.root_verify_key
    root_verify_key = queue_stash.db.root_verify_key
    config = queue_stash.db.config
    server_uid = queue_stash.db.server_uid

    def delete_in_new_thread(obj: QueueItem):
        queue_stash = QueueStash.random(
            root_verify_key=root_verify_key,
            config=config,
            server_uid=server_uid,
        )
        return queue_stash.delete_by_uid(root_verify_key, uid=obj.id)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                lambda obj: queue_stash.set(
                    root_verify_key,
                    mock_queue_object(),
                ),
                range(50),
            )
        )
        objs = [item.unwrap() for item in results]

        results = list(
            executor.map(
                delete_in_new_thread,
                objs,
            )
        )
        assert all(res.is_ok() for res in results), "Error occurred during execution"

    assert len(queue_stash) == 0
