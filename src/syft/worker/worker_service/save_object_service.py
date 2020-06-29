from syft.worker.worker_service import WorkerService, message_service_mapping
from syft.message import SaveObjectMessage
from syft.typecheck.typecheck import type_hints


class SaveObjectService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: "Worker", msg: SaveObjectMessage):
        # self.store.store_object(msg.id, msg.obj)
        pass

    @staticmethod
    @type_hints
    def message_type_handler() -> type:
        return SaveObjectMessage

    @staticmethod
    @type_hints
    def register_service() -> None:
        message_service_mapping[SaveObjectMessage] = SaveObjectService


SaveObjectService.register_service()
