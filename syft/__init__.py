r"""
PySyft is a Python library for secure, private Deep Learning.
PySyft decouples private data from model training, using Federated Learning,
Differential Privacy, and Multi-Party Computation (MPC) within PyTorch.
"""
# We load these modules first so that syft knows which are available
from syft import dependency_check
from syft import frameworks  # Triggers registration of any available frameworks

# Major imports
from syft.version import __version__

# This import statement is strictly here to trigger registration of syft
# tensor types inside hook_args.py.
import syft.frameworks.torch.hook.hook_args

import logging

logger = logging.getLogger(__name__)

# The purpose of the following import section is to increase the convenience of using
# PySyft by making it possible to import the most commonly used objects from syft
# directly (i.e., syft.TorchHook or syft.VirtualWorker or syft.LoggingTensor)

# Tensorflow / Keras dependencies
# Import Hooks

if dependency_check.tfe_available:
    from syft.frameworks.keras import KerasHook
    from syft.workers.tfe import TFECluster
    from syft.workers.tfe import TFEWorker

    __all__ = ["KerasHook", "TFECluster", "TFEWorker"]
else:
    logger.info("TF Encrypted Keras not available.")
    __all__ = []

# Pytorch dependencies
# Import Hook
from syft.frameworks.torch.hook.hook import TorchHook

# Import grids
from syft.grid import VirtualGrid

# Import sandbox
from syft.sandbox import create_sandbox

# Import federate learning objects
from syft.frameworks.torch.federated import FederatedDataset, FederatedDataLoader, BaseDataset
from syft.federated.train_config import TrainConfig

# Import messaging objects
from syft.messaging.plan import Plan
from syft.messaging.plan import func2plan
from syft.messaging.plan import method2plan
from syft.messaging.plan import make_plan

# Import Worker Types
from syft.workers.virtual import VirtualWorker
from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.websocket_server import WebsocketServerWorker

# Import Syft's Public Tensor Types
from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters.crt_precision import CRTPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.large_precision import LargePrecisionTensor
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.pointers.multi_pointer import MultiPointerTensor

# Import serialization tools
from syft import serde

# import functions
from syft.frameworks.torch.functions import combine_pointers

__all__.extend(
    [
        "frameworks",
        "serde",
        "TorchHook",
        "VirtualWorker",
        "WebsocketClientWorker",
        "WebsocketServerWorker",
        "Plan",
        "func2plan",
        "method2plan",
        "make_plan",
        "LoggingTensor",
        "AdditiveSharingTensor",
        "CRTPrecisionTensor",
        "AutogradTensor",
        "FixedPrecisionTensor",
        "LargePrecisionTensor",
        "PointerTensor",
        "MultiPointerTensor",
        "VirtualGrid",
        "create_sandbox",
        "combine_pointers",
        "FederatedDataset",
        "FederatedDataLoader",
        "BaseDataset",
        "TrainConfig",
    ]
)

local_worker = None
torch = None
framework = None

if "ID_PROVIDER" not in globals():
    from syft.generic.id_provider import IdProvider

    ID_PROVIDER = IdProvider()
