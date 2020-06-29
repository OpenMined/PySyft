from syft.store.store import ObjectStore
from syft.ast.globals import Globals
from syft.lib import supported_frameworks
from syft.typecheck.typecheck import type_hints
from syft.worker.worker_service import message_service_mapping


class Worker:
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

    @type_hints
    def __init__(self, id: str):
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

    def recv_msg(self, msg) -> None:
        # return self.msg_router[type(msg)](msg)
        pass

    def __repr__(self):
        return f"<Worker id:{self.id}>"
