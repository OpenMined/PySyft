from __future__ import annotations

from ...abstract.service import WorkerService
from ....message import RunFunctionOrConstructorMessage
from ..... import type_hints
from .....common import AbstractWorker
from typing import List

class RunFunctionOrConstructorService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: AbstractWorker, msg: RunFunctionOrConstructorMessage) -> None:
        pass

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [RunFunctionOrConstructorMessage]
