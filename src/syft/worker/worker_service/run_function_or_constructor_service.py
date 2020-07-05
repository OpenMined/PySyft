from __future__ import annotations

from .worker_service import WorkerService
from .. import message_service_mapping
from ...message import RunFunctionOrConstructorMessage
from ....typecheck import type_hints

class RunFunctionOrConstructorService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: "Worker", msg: RunFunctionOrConstructorMessage) -> None:
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
