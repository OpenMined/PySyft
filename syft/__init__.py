"""Some syft imports..."""
from . import core
from . import mpc
from torch.autograd import Variable
from torch.nn import Parameter
from torch.autograd import Variable as Var
from .core.frameworks.torch import TorchHook
from .core.frameworks.torch import _SyftTensor, _LocalTensor, _PointerTensor, _FixedPrecisionTensor, _PlusIsMinusTensor
from syft.core.workers import VirtualWorker, SocketWorker

__all__ = ['core', 'mpc']

import syft
import torch

for f in dir(torch):
    if("_" not in f):
        setattr(syft,f,getattr(torch,f))

setattr(syft, 'deser', _SyftTensor.deser)