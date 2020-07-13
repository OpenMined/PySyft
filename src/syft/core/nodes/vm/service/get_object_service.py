from __future__ import annotations

from ..... import type_hints
from .worker_service import WorkerService
from ....message import GetObjectMessage
from .....common import AbstractWorker


class GetObjectService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: AbstractWorker, msg: GetObjectMessage) -> object: #TODO: return StoreableObject
        return worker.store.get_object(msg.id)

    @staticmethod
    @type_hints
    def message_type_handler() -> type:
        return GetObjectMessage
