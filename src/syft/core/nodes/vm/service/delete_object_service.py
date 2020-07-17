from __future__ import annotations
from .....typecheck import type_hints
from ...abstract.service import WorkerService
from ....message import DeleteObjectMessage
from .....common import AbstractWorker
from typing import List

class DeleteObjectService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: AbstractWorker, msg: DeleteObjectMessage) -> None:
        return worker.store.delete_object(msg.id)

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [DeleteObjectMessage]
