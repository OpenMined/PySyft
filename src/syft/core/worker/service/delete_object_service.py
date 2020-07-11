from __future__ import annotations
import syft
from syft.worker.worker_service import WorkerService, message_service_mapping
from syft.message import DeleteObjectMessage


class DeleteObjectService(WorkerService):
    @staticmethod
    @syft.typecheck.type_hints
    def process(worker: "syft.worker.Worker", msg: DeleteObjectMessage) -> None:
        # return self.store.delete_object(msg.id)
        pass

    @staticmethod
    @syft.typecheck.type_hints
    def message_type_handler() -> type:
        return DeleteObjectMessage

    @staticmethod
    @syft.typecheck.type_hints
    def register_service() -> None:
        message_service_mapping[DeleteObjectMessage] = DeleteObjectService


DeleteObjectService.register_service()
