r"""
PySyft is a Python library for secure, private Deep Learning.
PySyft decouples private data from model training, using Federated Learning,
Differential Privacy, and Multi-Party Computation (MPC) within PyTorch.
"""
# Major imports
from syft import frameworks
from syft import workers
from syft import codes
from syft import federated
from .version import __version__

import logging

logger = logging.getLogger(__name__)

# The purpose of the following import section is to increase the convenience of using
# PySyft by making it possible to import the most commonly used objects from syft
# directly (i.e., syft.TorchHook or syft.VirtualWorker or syft.LoggingTensor)

# Tensorflow / Keras dependencies
# Import Hooks

from syft import dependency_check

if dependency_check.tfe_available:
    from syft.frameworks.keras import KerasHook
    from syft.workers import TFECluster
    from syft.workers import TFEWorker
else:
    logger.info("TF Encrypted Keras not available.")

# Pytorch dependencies
# Import Hook
from syft.frameworks.torch import TorchHook

# Import Tensor Types
from syft.frameworks.torch.tensors.decorators import LoggingTensor
from syft.frameworks.torch.tensors.interpreters import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters import AutogradTensor
from syft.generic.pointers import MultiPointerTensor
from syft.generic.pointers import PointerTensor

# import other useful classes
from syft.frameworks.torch.federated import FederatedDataset, FederatedDataLoader, BaseDataset

# Import grids
from syft.grid import VirtualGrid

# Import sandbox
from syft.sandbox import create_sandbox

# Import federate learning objects
from syft.federated import TrainConfig

# Import messaging objects
from syft.messaging.message import Message
from syft.messaging import Plan
from syft.messaging import func2plan
from syft.messaging import method2plan
from syft.messaging import make_plan

# Import Worker Types
from syft.workers import VirtualWorker


# Import Tensor Types
from syft.frameworks.torch.tensors.decorators import LoggingTensor
from syft.frameworks.torch.tensors.interpreters import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters import CRTPrecisionTensor
from syft.frameworks.torch.tensors.interpreters import AutogradTensor
from syft.frameworks.torch.tensors.interpreters import FixedPrecisionTensor
from syft.frameworks.torch.tensors.interpreters import LargePrecisionTensor

from syft.generic.pointers import ObjectPointer
from syft.generic.pointers import CallablePointer
from syft.generic.pointers import ObjectWrapper

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
    from syft.generic import IdProvider

    ID_PROVIDER = IdProvider()
