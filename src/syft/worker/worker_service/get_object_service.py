from __future__ import annotations

from ....typecheck import type_hints
from .worker_service import WorkerService
from .. import message_service_mapping
from ...message import GetObjectMessage
from ..worker import Worker

class GetObjectService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: Worker, msg: GetObjectMessage) -> None:
        pass

    @staticmethod
    @type_hints
    def message_type_handler() -> type:
        return GetObjectMessage

    @staticmethod
    @type_hints
    def register_service() -> None:
        message_service_mapping[GetObjectMessage] = GetObjectService


GetObjectService.register_service()
