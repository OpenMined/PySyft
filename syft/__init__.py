r"""
PySyft is a Python library for secure, private Deep Learning.
PySyft decouples private data from model training, using Federated Learning,
Differential Privacy, and Multi-Party Computation (MPC) within PyTorch.
"""
# We load these modules first so that syft knows which are available
from syft import dependency_check
from syft import frameworks  # Triggers registration of any available frameworks # noqa:F401

# Major imports

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
    from syft.frameworks.keras import KerasHook  # noqa: F401
    from syft.workers.tfe import TFECluster  # noqa: F401
    from syft.workers.tfe import TFEWorker  # noqa: F401

    __all__.extend(["KerasHook", "TFECluster", "TFEWorker"])
else:
    logger.info("TF Encrypted Keras not available.")

# Pytorch dependencies
# Import Hook
from syft.frameworks.torch.hook.hook import TorchHook  # noqa: E402,F401

# Import grids
from syft.grid.private_grid import PrivateGridNetwork  # noqa: E402,F401
from syft.grid.public_grid import PublicGridNetwork  # noqa: E402,F401


# Import sandbox
from syft.sandbox import create_sandbox, make_hook  # noqa: E402,F401

# Import federate learning objects
from syft.frameworks.torch.fl import (  # noqa: E402, F401
    FederatedDataset,
    FederatedDataLoader,
    BaseDataset,
)

# Import messaging objects
from syft.execution.protocol import Protocol  # noqa: E402, F401
from syft.execution.protocol import func2protocol  # noqa: E402, F401
from syft.execution.plan import Plan  # noqa: E402, F401
from syft.execution.plan import func2plan  # noqa: E402, F401

# Import Worker Types
from syft.workers.virtual import VirtualWorker  # noqa: E402,F401
from syft.workers.websocket_client import WebsocketClientWorker  # noqa: E402,F401
from syft.workers.websocket_server import WebsocketServerWorker  # noqa: E402,F401

# Register PRZS such that all workers have the allowed commands from the przs file
import syft.frameworks.torch.mpc.przs  # noqa: E402,F401

# Import Syft's Public Tensor Types
from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor  # noqa: E402,F401
from syft.frameworks.torch.tensors.interpreters.additive_shared import (  # noqa: E402,F401
    AdditiveSharingTensor,
)
from syft.frameworks.torch.tensors.interpreters.replicated_shared import (  # noqa: E402,F401
    ReplicatedSharingTensor,
)
from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor  # noqa: E402,F401
from syft.frameworks.torch.tensors.interpreters.precision import (  # noqa: E402,F401
    FixedPrecisionTensor,
)
from syft.frameworks.torch.tensors.interpreters.numpy import (  # noqa: E402,F401
    create_numpy_tensor as NumpyTensor,
)

from syft.frameworks.torch.tensors.interpreters.private import PrivateTensor  # noqa: E402, F401
from syft.execution.placeholder import PlaceHolder  # noqa: E402, F401
from syft.generic.pointers.pointer_plan import PointerPlan  # noqa: E402, F401
from syft.generic.pointers.pointer_tensor import PointerTensor  # noqa: E402, F401
from syft.generic.pointers.multi_pointer import MultiPointerTensor  # noqa: E402, F401

# Import serialization tools
from syft import serde  # noqa: E402, F401

# import functions
from syft.frameworks.torch.functions import combine_pointers  # noqa: E402, F401
from syft.frameworks.torch.he.paillier import keygen  # noqa: E402, F401

# import common
import syft.common.util  # noqa: E402, F401


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
        "ReplicatedSharingTensor",
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
    ]
)

local_worker = None
torch = None
framework = None

if "ID_PROVIDER" not in globals():
    from syft.generic.id_provider import IdProvider

    ID_PROVIDER = IdProvider()
