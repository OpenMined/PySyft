from __future__ import annotations
from ....decorators import syft_decorator
from ...message import SyftMessage
from ....common import AbstractWorker
from typing import List


class WorkerService:
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(worker: AbstractWorker, msg: SyftMessage) -> object:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        raise NotImplementedError
