"""Some syft imports..."""
from . import frameworks
from . import workers
from . import serde
from . import codes

from syft.frameworks.torch import TorchHook
from syft.workers import VirtualWorker

__all__ = ["frameworks", "workers", "serde", "TorchHook", "VirtualWorker", "codes"]

local_worker = None
torch = None
