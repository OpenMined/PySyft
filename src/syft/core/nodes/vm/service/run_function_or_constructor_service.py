from __future__ import annotations

from .worker_service import WorkerService
from ....message import RunFunctionOrConstructorMessage
from ..... import type_hints
from .....common import AbstractWorker


class RunFunctionOrConstructorService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: AbstractWorker, msg: RunFunctionOrConstructorMessage) -> None:
        pass

    @staticmethod
    @type_hints
    def message_type_handler() -> type:
        return RunFunctionOrConstructorMessage
