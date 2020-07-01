from __future__ import annotations
import syft
from syft.worker.worker_service import WorkerService, message_service_mapping
from syft.message import GetObjectMessage


class GetObjectService(WorkerService):
    @staticmethod
    @syft.typecheck.type_hints
    def process(worker: "syft.worker.Worker", msg: GetObjectMessage) -> None:
        pass

    @staticmethod
    @syft.typecheck.type_hints
    def message_type_handler() -> type:
        return GetObjectMessage

    @staticmethod
    @syft.typecheck.type_hints
    def register_service() -> None:
        message_service_mapping[GetObjectMessage] = GetObjectService


GetObjectService.register_service()
