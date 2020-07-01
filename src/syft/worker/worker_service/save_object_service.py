from __future__ import annotations
import syft
from syft.worker.worker_service import WorkerService, message_service_mapping
from syft.message import SaveObjectMessage


class SaveObjectService(WorkerService):
    @staticmethod
    @syft.typecheck.type_hints
    def process(worker: "syft.worker.Worker", msg: SaveObjectMessage):
        # self.store.store_object(msg.id, msg.obj)
        pass

    @staticmethod
    @syft.typecheck.type_hints
    def message_type_handler() -> type:
        return SaveObjectMessage

    @staticmethod
    @syft.typecheck.type_hints
    def register_service() -> None:
        message_service_mapping[SaveObjectMessage] = SaveObjectService


SaveObjectService.register_service()
