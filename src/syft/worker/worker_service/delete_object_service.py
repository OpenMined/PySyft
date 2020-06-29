from syft.worker.worker_service import WorkerService, message_service_mapping
from syft.message import DeleteObjectMessage
from syft.typecheck.typecheck import type_hints


class DeleteObjectService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: "Worker", msg: DeleteObjectMessage) -> None:
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
