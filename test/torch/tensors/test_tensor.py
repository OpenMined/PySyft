import random

import torch
import syft

from syft.frameworks.torch.tensors.interpreters import PointerTensor


def test_init():
    hook = syft.TorchHook(torch, verbose=True)
    tensor_extension = torch.Tensor()
    assert tensor_extension.id is not None
    assert tensor_extension.owner is not None
