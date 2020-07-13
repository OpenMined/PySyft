from __future__ import annotations

from ..... import type_hints
from .worker_service import WorkerService
from ....message import ReprMessage
from ....message import ReprReplyMessage
from .....common import AbstractWorker


class ReprService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: AbstractWorker, msg: ReprMessage) -> ReprReplyMessage:
        return ReprReplyMessage(value=worker.__repr__(), route=None)

    @staticmethod
    @type_hints
    def message_type_handler() -> type:
        return ReprMessage
