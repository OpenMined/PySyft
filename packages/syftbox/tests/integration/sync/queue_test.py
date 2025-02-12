import random
from pathlib import Path
from queue import Empty

import pytest
from pydantic import BaseModel

from syftbox.client.plugins.sync.queue import SyncQueue, SyncQueueItem


class MockFileChangeInfo(BaseModel):  # noqa: F821
    path: Path


def test_sync_queue():
    queue = SyncQueue()

    n = 10

    paths = [Path(f"file_{i}.txt") for i in range(n)]
    priorities = [random.uniform(0, 1000) for _ in range(n)]
    priorities[0] = int(priorities[0])  # int and float should both work
    items = [SyncQueueItem(priority, MockFileChangeInfo(path=path)) for path, priority in zip(paths, priorities)]
    items_sorted = sorted(items, key=lambda x: x.priority)

    for item in items:
        queue.put(item)

    assert not queue.empty()
    assert set(queue.all_items.keys()) == set(paths)

    for item in items_sorted:
        assert queue.get() == item

    assert queue.empty()
    assert len(queue.all_items) == 0
    with pytest.raises(Empty):
        queue.get(block=False)


def test_sync_queue_dedupe():
    queue = SyncQueue()

    path = Path("file.txt")

    queue.put(SyncQueueItem(1, MockFileChangeInfo(path=path)))
    assert set(queue.all_items.keys()) == {path}
    assert not queue.empty()

    for _ in range(10):
        queue.put(SyncQueueItem(random.random(), MockFileChangeInfo(path=path)))

    assert set(queue.all_items.keys()) == {path}

    queue.get()
    assert len(queue.all_items) == 0
    assert queue.empty()
