import pytest

import torch

from syft.frameworks.torch.pointers import PointerTensor
from syft.exceptions import InvalidTensorForRemoteGet


def test___str__(workers):
    bob = workers["bob"]
    tensor = torch.Tensor([1, 2, 3, 4])
    assert isinstance(tensor.__str__(), str)

    tensor_ptr = tensor.send(bob)
    assert isinstance(tensor_ptr.__str__(), str)


def test___repr__(workers):
    bob = workers["bob"]

    tensor = torch.Tensor([1, 2, 3, 4])
    assert isinstance(tensor.__repr__(), str)

    tensor_ptr = tensor.send(bob)
    assert isinstance(tensor_ptr.__repr__(), str)

    tensor = torch.Tensor([1, 2, 3, 4]).tag("#my_tag").describe("This is a description")
    assert isinstance(tensor.__repr__(), str)


def test_overload_reshape():
    tensor = torch.Tensor([1, 2, 3, 4])
    tensor_reshaped = tensor.reshape((2, 2))
    tensor_matrix = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert (tensor_reshaped == tensor_matrix).all()


def test_owner_default(hook):
    tensor = torch.Tensor([1, 2, 3, 4, 5])
    assert tensor.owner == hook.local_worker


def test_create_pointer(hook, workers):
    bob = workers["bob"]

    tensor = torch.Tensor([1, 2, 3, 4, 5])

    ptr = tensor.create_pointer(
        location=bob, id_at_location=1, register=False, owner=hook.local_worker, ptr_id=2
    )

    assert ptr.owner == hook.local_worker
    assert ptr.location == bob
    assert ptr.id_at_location == 1
    assert ptr.id == 2

    ptr2 = tensor.create_pointer(owner=hook.local_worker)
    assert isinstance(ptr2.__str__(), str)
    assert isinstance(ptr2.__repr__(), str)


def test_create_pointer_defaults(workers):
    bob = workers["bob"]

    tensor = torch.Tensor([1, 2, 3, 4, 5])

    ptr = tensor.create_pointer(location=bob)

    assert ptr.owner == tensor.owner
    assert ptr.location == bob


def test_get(workers):
    bob = workers["bob"]

    tensor = torch.rand(5, 3)
    pointer = tensor.send(bob)

    assert type(pointer.child) == PointerTensor
    assert (pointer.get() == tensor).all()


def test_invalid_remote_get(workers):
    bob = workers["bob"]

    tensor = torch.rand(5, 3)
    pointer = tensor.send(bob)
    with pytest.raises(InvalidTensorForRemoteGet):
        pointer.remote_get()


def test_remote_get(hook, workers):
    me = workers["me"]
    bob = workers["bob"]
    alice = workers["alice"]

    x = torch.tensor([1, 2, 3, 4, 5])
    ptr_ptr_x = x.send(bob).send(alice)

    assert ptr_ptr_x.owner == me
    assert ptr_ptr_x.location == alice
    assert x.id in bob._objects

    assert len(bob._objects) == 1
    assert len(alice._objects) == 1

    ptr_ptr_x.remote_get()

    assert len(bob._objects) == 0
    assert len(alice._objects) == 1


def test_copy():
    tensor = torch.rand(5, 3)
    coppied_tensor = tensor.copy()
    assert (tensor == coppied_tensor).all()
    assert tensor is not coppied_tensor


def test_size():
    tensor = torch.rand(5, 3)
    assert tensor.size() == torch.Size([5, 3])
    assert tensor.size() == tensor.shape
    assert tensor.size(0) == tensor.shape[0]


# Compare local dim with the remote one
def test_dim(workers):
    tensor_local = torch.randn(5, 3)
    tensor_remote = tensor_local.send(workers["alice"])

    assert tensor_local.dim() == tensor_remote.dim()


def test_does_not_require_large_precision():
    x = torch.tensor([[[-1.5, 2.0, 30000000000.0]], [[4.5, 5.0, 6.0]], [[7.0, 8.0, 9.0]]])
    base = 10
    prec_fractional = 3
    max_precision = 62
    assert not x._requires_large_precision(max_precision, base, prec_fractional)


def test_requires_large_precision():
    x = torch.tensor([[[-1.5, 2.0, 30000000000.0]], [[4.5, 5.0, 6.0]], [[7.0, 8.0, 9.0]]])
    base = 10
    prec_fractional = 256
    max_precision = 62
    assert x._requires_large_precision(max_precision, base, prec_fractional)


def test_roll(workers):
    x = torch.tensor([1.0, 2.0, 3, 4, 5])
    expected = torch.roll(x, -1)

    index = torch.tensor([-1.0])
    result = torch.roll(x, index)

    assert (result == expected).all()
