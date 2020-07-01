from __future__ import annotations
import syft


class WorkerService:
    @staticmethod
    @syft.typecheck.type_hints
    def process(worker: "syft.worker.Worker", msg: "syft.message.SyftMessage") -> None:
        raise NotImplementedError

    @staticmethod
    @syft.typecheck.type_hints
    def message_type_handler() -> "syft.message.SyftMessage":
        raise NotImplementedError

    @staticmethod
    @syft.typecheck.type_hints
    def register_service() -> None:
        raise NotImplementedError
