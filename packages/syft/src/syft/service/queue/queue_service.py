# stdlib

# relative
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ..service import AbstractService
from .queue_stash import QueueStash


@serializable(canonical_name="QueueService", version=1)
class QueueService(AbstractService):
    stash: QueueStash

    def __init__(self, store: DBManager) -> None:
        self.stash = QueueStash(store=store)
