from __future__ import annotations
from syft.typecheck.typecheck import type_hints


class WorkerService:
    @staticmethod
    @type_hints
    def process(worker: "Worker", msg: "SyftMessage") -> None:
        raise NotImplementedError

    @staticmethod
    @type_hints
    def message_type_handler() -> "SyftMessage":
        return None

    @staticmethod
    @type_hints
    def register_service() -> None:
        raise NotImplementedError
