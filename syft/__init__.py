"""Some syft imports..."""
from . import frameworks
from . import workers
from . import serde
from . import codes

# CONVENIENCE HOOKS
# The purpose of the following import section is to increase the convenience of using
# PySyft by making it possible to import the most commonly used objects from syft
# directly (i.e., syft.TorchHook or syft.VirtualWorker or syft.LoggingTensor)

# Import Hook
from syft.frameworks.torch import TorchHook

# Import Worker Types
from syft.workers import VirtualWorker

# Import Tensor Types
from syft.frameworks.torch.tensors import LoggingTensor
from syft.frameworks.torch.tensors import PointerTensor

# import modules
from syft.frameworks.torch import optim

__all__ = [
    "frameworks",
    "workers",
    "serde",
    "TorchHook",
    "VirtualWorker",
    "codes",
    "LoggingTensor",
    "PointerTensor",
    "optim",
]

local_worker = None
torch = None
