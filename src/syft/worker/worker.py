from __future__ import annotations
import syft
from syft.store.store import ObjectStore
from syft.ast.globals import Globals
from syft.lib import supported_frameworks
from syft.worker.worker_service import message_service_mapping
from syft.worker.worker_supervisor.supervisor import WorkerSupervisor
from syft.worker.worker_supervisor.stats import WorkerStats


class Worker(metaclass=WorkerSupervisor):
    """
    Basic class for a syft worker behavior, explicit purpose workers will
    inherit this class (eg. WebsocketWorker, VirtualWorker).

    A worker is a collection of objects owned by a machine, a list of supported
    frameworks used for remote execution and a message router. The objects
    owned by the worker are placed in an ObjectStore object, the list of
    frameworks are a list of Globals and the message router is a dict that maps
    a message type to a processing method.

    Each worker is identified by an id of type str.
    """

    @syft.typecheck.type_hints
    def __init__(self, id: str, debug: bool = False):
        self.id = id
        self.store = ObjectStore()
        self.frameworks = Globals()
        for fw in supported_frameworks:
            for name, ast in fw.ast.attrs.items():
                if name in self.frameworks.attrs:
                    raise KeyError(
                        "Framework already imported. Why are you importing it twice?"
                    )
                self.frameworks.attrs[name] = ast

        self.msg_router = message_service_mapping
        self.worker_stats = None
        if debug:
            self.worker_stats = WorkerStats()

    @syft.typecheck.type_hints
    def recv_msg(self, msg: "syft.message.SyftMessage") -> None:
        pass

    @syft.typecheck.type_hints
    def _send_msg(self) -> None:
        raise NotImplementedError

    @syft.typecheck.type_hints
    def _recv_msg(self) -> None:
        raise NotImplementedError

    def __repr__(self):
        if self.worker_stats:
            return f"Worker: {self.id}\n{self.worker_stats}"

        return f"Worker id:{self.id}"
