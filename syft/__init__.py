r"""
PySyft is a Python library for secure, private Deep Learning.
PySyft decouples private data from model training, using Federated Learning,
Differential Privacy, and Multi-Party Computation (MPC) within PyTorch.
"""
# We load this module first so that syft knows which frameworks are available
from syft import dependency_check

# Major imports
from syft import frameworks
from syft import codes
from syft.version import __version__

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
else:
    logger.info("TF Encrypted Keras not available.")

# Pytorch dependencies
# Import Hook
from syft.frameworks.torch.hook.hook import TorchHook

# Import Tensor Types
from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from syft.generic.pointers.multi_pointer import MultiPointerTensor
from syft.generic.pointers.pointer_tensor import PointerTensor

# import other useful classes
from syft.frameworks.torch.federated import FederatedDataset, FederatedDataLoader, BaseDataset

# Import grids
from syft.grid import VirtualGrid

# Import sandbox
from syft.sandbox import create_sandbox

# Import federate learning objects
from syft.federated.train_config import TrainConfig

# Import messaging objects
from syft.messaging.message import Message
from syft.messaging.plan import Plan
from syft.messaging.plan import func2plan
from syft.messaging.plan import method2plan
from syft.messaging.plan import make_plan

# Import Worker Types
from syft.workers.virtual import VirtualWorker


# Import Tensor Types
from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters.crt_precision import CRTPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.large_precision import LargePrecisionTensor

from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.pointers.callable_pointer import CallablePointer
from syft.generic.pointers.object_wrapper import ObjectWrapper

# Import serialization tools
from syft import serde
from syft.serde import torch_serde

# import functions
from syft.frameworks.torch.functions import combine_pointers

__all__ = [
    "frameworks",
    "workers",
    "serde",
    "torch_serde",
    "TorchHook",
    "VirtualWorker",
    "Plan",
    "codes",
    "LoggingTensor",
    "PointerTensor",
    "VirtualGrid",
    "ObjectWrapper",
    "LargePrecisionTensor",
    "create_sandbox",
]

local_worker = None
torch = None

if "ID_PROVIDER" not in globals():
    from syft.generic.id_provider import IdProvider

    ID_PROVIDER = IdProvider()
