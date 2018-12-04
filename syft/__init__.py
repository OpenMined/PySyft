"""Some syft imports..."""
from . import frameworks
from . import workers
from . import serde

from syft.frameworks.torch import TorchHook
from syft.workers import VirtualWorker

__all__ = ["frameworks", "workers", "serde", "TorchHook", "VirtualWorker"]

local_worker = None
torch = None
