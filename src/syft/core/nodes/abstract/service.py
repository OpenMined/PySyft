from __future__ import annotations
from .... import type_hints
from ...message import SyftMessage
from ....common import AbstractWorker
from typing import List


class WorkerService:
    @staticmethod
    @type_hints
    def process(worker: AbstractWorker, msg: SyftMessage) -> object:
        raise NotImplementedError

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        raise NotImplementedError
