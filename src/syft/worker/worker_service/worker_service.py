from __future__ import annotations
from .... import type_hints
from ..worker import Worker
from ...message import SyftMessage


class WorkerService:
    @staticmethod
    @type_hints
    def process(worker: Worker, msg: SyftMessage) -> None:
        raise NotImplementedError

    @staticmethod
    @type_hints
    def message_type_handler() -> SyftMessage:
        raise NotImplementedError

    @staticmethod
    @type_hints
    def register_service() -> None:
        raise NotImplementedError
