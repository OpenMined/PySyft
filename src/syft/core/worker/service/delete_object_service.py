from __future__ import annotations
from ....typecheck import type_hints
from .worker_service import WorkerService
from .. import message_service_mapping
from ...message import DeleteObjectMessage
from ..worker import Worker

class DeleteObjectService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: Worker, msg: DeleteObjectMessage) -> None:
        # return self.store.delete_object(msg.id)
        pass

    @staticmethod
    @type_hints
    def message_type_handler() -> type:
        return DeleteObjectMessage

    @staticmethod
    @type_hints
    def register_service() -> None:
        message_service_mapping[DeleteObjectMessage] = DeleteObjectService


DeleteObjectService.register_service()
