from __future__ import annotations
from .... import type_hints
from .worker_service import WorkerService
from .. import message_service_mapping
from ...message import DeleteObjectMessage
from ....common import AbstractWorker


class DeleteObjectService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: AbstractWorker, msg: DeleteObjectMessage) -> None:
        return worker.store.delete_object(msg.id)

    @staticmethod
    @type_hints
    def message_type_handler() -> type:
        return DeleteObjectMessage

    @staticmethod
    @type_hints
    def register_service() -> None:
        message_service_mapping[DeleteObjectMessage] = DeleteObjectService


DeleteObjectService.register_service()
