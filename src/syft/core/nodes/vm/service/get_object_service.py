from __future__ import annotations

from ..... import type_hints
from ...abstract.service import WorkerService
from ....message import GetObjectMessage
from .....common import AbstractWorker
from typing import List


class GetObjectService(WorkerService):
    @staticmethod
    @type_hints
    def process(
        worker: AbstractWorker, msg: GetObjectMessage
    ) -> object:  # TODO: return StoreableObject
        return worker.store.get_object(msg.id)

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [GetObjectMessage]
