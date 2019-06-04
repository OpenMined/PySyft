import torch
import syft


def test_init():
    hook = syft.TorchHook(torch, verbose=True)
    tensor_extension = torch.Tensor()
    assert tensor_extension.id is not None
    assert tensor_extension.owner is not None
