from syft.worker.worker_service import WorkerService, message_service_mapping
from syft.message import RunFunctionOrConstructorMessage
from syft.typecheck.typecheck import type_hints


class RunFunctionOrConstructorService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: "Worker", msg: RunFunctionOrConstructorMessage):
        pass

    @staticmethod
    @type_hints
    def message_type_handler() -> type:
        return RunFunctionOrConstructorMessage

    @staticmethod
    @type_hints
    def register_service() -> None:
        message_service_mapping[
            RunFunctionOrConstructorMessage
        ] = RunFunctionOrConstructorService


RunFunctionOrConstructorService.register_service()
