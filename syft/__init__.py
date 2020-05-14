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

__all__ = []
if dependency_check.tfe_available:
    from syft.frameworks.keras import KerasHook
    from syft.workers.tfe import TFECluster
    from syft.workers.tfe import TFEWorker

    __all__.extend(["KerasHook", "TFECluster", "TFEWorker"])
else:
    logger.info("TF Encrypted Keras not available.")

# Pytorch dependencies
# Import Hook
from syft.frameworks.torch.hook.hook import TorchHook

# Import grids
from syft.grid.private_grid import PrivateGridNetwork
from syft.grid.public_grid import PublicGridNetwork

# Import sandbox
from syft.sandbox import create_sandbox, make_hook

# Import federate learning objects
from syft.frameworks.torch.fl import FederatedDataset, FederatedDataLoader, BaseDataset
from syft.federated.train_config import TrainConfig

# Import messaging objects
from syft.execution.protocol import Protocol
from syft.execution.protocol import func2protocol
from syft.execution.plan import Plan
from syft.execution.plan import func2plan

# Import Worker Types
from syft.workers.virtual import VirtualWorker
from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.websocket_server import WebsocketServerWorker

# Import Syft's Public Tensor Types
from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.numpy import create_numpy_tensor as NumpyTensor
from syft.frameworks.torch.tensors.interpreters.private import PrivateTensor
from syft.execution.placeholder import PlaceHolder
from syft.generic.pointers.pointer_plan import PointerPlan
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.pointers.multi_pointer import MultiPointerTensor

# Import serialization tools
from syft import serde

# import functions
from syft.frameworks.torch.functions import combine_pointers
from syft.frameworks.torch.he.paillier import keygen

# import common
import syft.common.util


def pool():
    if not hasattr(syft, "_pool"):
        import multiprocessing

        syft._pool = multiprocessing.Pool()
    return syft._pool


__all__.extend(
    [
        "frameworks",
        "serde",
        "TorchHook",
        "VirtualWorker",
        "WebsocketClientWorker",
        "WebsocketServerWorker",
        "Protocol",
        "func2protocol",
        "Plan",
        "func2plan",
        "make_plan",
        "LoggingTensor",
        "AdditiveSharingTensor",
        "AutogradTensor",
        "FixedPrecisionTensor",
        "PointerTensor",
        "MultiPointerTensor",
        "PrivateGridNetwork",
        "PublicGridNetwork",
        "create_sandbox",
        "make_hook",
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
