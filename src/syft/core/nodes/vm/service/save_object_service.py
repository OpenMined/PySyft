from __future__ import annotations

from ...abstract.service import WorkerService
from ....message import SaveObjectMessage

from ..... import type_hints

from .....common import AbstractWorker
from typing import List

class SaveObjectService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: AbstractWorker, msg: SaveObjectMessage) -> None:
        worker.store.store_object(msg.id, msg.obj)
        pass

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [SaveObjectMessage]
