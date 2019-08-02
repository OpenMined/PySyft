from syft.workers.abstract import AbstractWorker  # noqa: F401
from syft.workers.base import BaseWorker  # noqa: F401
from syft.workers.virtual import VirtualWorker  # noqa: F401

from syft.workers.websocket_client import WebsocketClientWorker  # noqa: F401
from syft.workers.websocket_server import WebsocketServerWorker  # noqa: F401

from syft import dependency_check

__all__ = ["base", "virtual", "websocket_client", "socketio_client", "BaseWorker"]

if dependency_check.keras_available:
    from syft.workers.tfe import TFECluster  # noqa: F401
    from syft.workers.tfe import TFEWorker  # noqa: F401

    __all__.append("tfe")
