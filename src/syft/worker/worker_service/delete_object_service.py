from __future__ import annotations
from .....typecheck import type_hints
from .worker_service import WorkerService
from ....message import DeleteObjectMessage
from .....common import AbstractWorker


class DeleteObjectService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: AbstractWorker, msg: DeleteObjectMessage) -> None:
        return worker.store.delete_object(msg.id)

    @staticmethod
    @type_hints
    def message_type_handler() -> type:
        return DeleteObjectMessage

