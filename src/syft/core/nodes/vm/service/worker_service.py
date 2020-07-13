from __future__ import annotations
from ..... import type_hints

from ....message import SyftMessage

from .....common import AbstractWorker


class WorkerService:
    @staticmethod
    @type_hints
    def process(worker: AbstractWorker, msg: SyftMessage) -> object:
        raise NotImplementedError

    @staticmethod
    @type_hints
    def message_type_handler() -> SyftMessage:
        raise NotImplementedError
