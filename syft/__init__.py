"""Some syft imports..."""
from . import frameworks
from . import workers
from . import serde

from syft.frameworks.torch import TorchHook
from syft.workers import VirtualWorker

from torch import Tensor

__all__ = ["frameworks", "workers", "serde"]
