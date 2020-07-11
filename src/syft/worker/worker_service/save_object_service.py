from __future__ import annotations

from .worker_service import WorkerService
from .. import message_service_mapping
from ...message import SaveObjectMessage

from .... import type_hints

from ..worker import Worker


class SaveObjectService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: Worker, msg: SaveObjectMessage) -> None:
        # self.store.store_object(msg.id, msg.obj)
        pass

    @staticmethod
    @type_hints
    def message_type_handler() -> type:
        return SaveObjectMessage

    @staticmethod
    @type_hints
    def register_service() -> None:
        message_service_mapping[SaveObjectMessage] = SaveObjectService


SaveObjectService.register_service()
