"""Some syft imports..."""
from syft import core
from syft import spdz
from syft.core.frameworks.torch import _SyftTensor

from torch.autograd import Variable
from torch.nn import Parameter
from torch.autograd import Variable as Var
from syft.core.frameworks.torch import TorchHook
from syft.core.frameworks.torch import (
    _LocalTensor,
    _PointerTensor,
    _FixedPrecisionTensor,
    _PlusIsMinusTensor,
    _GeneralizedPointerTensor,
    _SPDZTensor,
    _SNNTensor,
)
from syft.core.workers import VirtualWorker, SocketWorker
from syft.core.frameworks.numpy import array

__all__ = ["core", "spdz"]

import syft
import torch

for f in dir(torch):
    if "_" not in f:
        setattr(syft, f, getattr(torch, f))

setattr(syft, "deser", _SyftTensor.deser)


# TODO: figure out how to let this be hooked here so that it happens
# automatically when you import syft. Right now it breaks if you accidentally
# hook again or if you need to hook it with a special local_worker (such as SocketWorker)
# hook = TorchHook(verbose=False)
