import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import PriorityQueue
from typing import Dict, Optional

from syftbox.client.plugins.sync.types import FileChangeInfo


@dataclass(order=True)
class SyncQueueItem:
    priority: int
    data: FileChangeInfo
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


class SyncQueue:
    """
    A thread-safe priority queue that supports deduplication based on the data field.

    Adding an item to the queue that already exists will be ignored, even if the priority is different.
    """

    def __init__(self, maxsize: int = 0):
        self.queue: PriorityQueue[SyncQueueItem] = PriorityQueue(maxsize=maxsize)
        self.all_items: Dict[Path, SyncQueueItem] = {}

        self.lock = threading.Lock()

    def put(self, item: SyncQueueItem, block: bool = False, timeout: Optional[float] = None) -> None:
        with self.lock:
            if item.data.path not in self.all_items:
                self.queue.put(item, block=block, timeout=timeout)
                self.all_items[item.data.path] = item

    def get(self, block: bool = False, timeout: Optional[float] = None) -> SyncQueueItem:
        with self.lock:
            item: SyncQueueItem = self.queue.get(block=block, timeout=timeout)
            self.all_items.pop(item.data.path, None)
            return item

    def empty(self) -> bool:
        return self.queue.empty()
