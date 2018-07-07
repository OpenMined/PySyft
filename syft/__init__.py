"""Some syft imports..."""
from . import core
from . import mpc
from .core.frameworks.torch import TorchHook
from .core.frameworks.torch import _AbstractTensor, _LocalTensor, _PointerTensor, _FixedPrecisionTensor
from syft.core.workers import VirtualWorker, SocketWorker

__all__ = ['core', 'mpc']

import syft
import torch

for f in dir(torch):
    setattr(syft,f,getattr(torch,f))

setattr(syft, 'deser', _AbstractTensor.deser)