from syft.worker.worker_service import WorkerService, message_service_mapping
from syft.message import GetObjectMessage
from syft.typecheck.typecheck import type_hints


class GetObjectService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: "Worker", msg: GetObjectMessage) -> None:
        pass

    @staticmethod
    @type_hints
    def message_type_handler() -> type:
        return GetObjectMessage

    @staticmethod
    @type_hints
    def register_service() -> None:
        message_service_mapping[GetObjectMessage] = GetObjectService


GetObjectService.register_service()
