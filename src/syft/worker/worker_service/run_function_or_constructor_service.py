from __future__ import annotations
import syft
from syft.worker.worker_service import WorkerService, message_service_mapping
from syft.message import RunFunctionOrConstructorMessage


class RunFunctionOrConstructorService(WorkerService):
    @staticmethod
    @syft.typecheck.type_hints
    def process(worker: "Worker", msg: RunFunctionOrConstructorMessage):
        pass

    @staticmethod
    @syft.typecheck.type_hints
    def message_type_handler() -> type:
        return RunFunctionOrConstructorMessage

    @staticmethod
    @syft.typecheck.type_hints
    def register_service() -> None:
        message_service_mapping[
            RunFunctionOrConstructorMessage
        ] = RunFunctionOrConstructorService


RunFunctionOrConstructorService.register_service()
