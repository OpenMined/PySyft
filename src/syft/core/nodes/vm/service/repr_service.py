from __future__ import annotations
from ..message.repr_message import ReprMessage
from ..message.repr_message import ReprReplyMessage
from ..... import type_hints
from ...abstract.service import WorkerService
from .....common import AbstractWorker
from typing import List

class ReprService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: AbstractWorker, msg: ReprMessage) -> ReprReplyMessage:
        return ReprReplyMessage(value=worker.__repr__(), route=None)

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [ReprMessage]
