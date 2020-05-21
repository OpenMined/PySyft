import pytest
import torch
from torch import nn
import torch.nn.functional as F
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.exceptions import InvalidTensorForRemoteGet
import syft


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
    assert x.id in bob.object_store._objects

    assert len(bob.object_store._tensors) == 1
    assert len(alice.object_store._tensors) == 1

    ptr_ptr_x.remote_get()

    assert len(bob.object_store._tensors) == 0
    assert len(alice.object_store._tensors) == 1


def test_remote_send(hook, workers):
    me = workers["me"]
    bob = workers["bob"]
    alice = workers["alice"]

    x = torch.tensor([1, 2, 3, 4, 5])
    # Note: behavior has been changed to point to the last pointer
    ptr_ptr_x = x.send(bob).remote_send(alice)

    assert ptr_ptr_x.owner == me
    assert ptr_ptr_x.location == bob
    assert x.id in alice.object_store._objects


def test_copy():
    tensor = torch.rand(5, 3)
    copied_tensor = tensor.copy()
    assert (tensor == copied_tensor).all()
    assert tensor is not copied_tensor


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


def test_roll(workers):
    x = torch.tensor([1.0, 2.0, 3, 4, 5])
    expected = torch.roll(x, -1)

    index = torch.tensor([-1.0])
    result = torch.roll(x, index)

    assert (result == expected).all()


def test_complex_model(workers):
    hook = syft.TorchHook(torch)
    bob = workers["bob"]
    tensor_local = torch.rand(4, 1, 32, 32)
    tensor_remote = tensor_local.send(bob)

    ## Instantiating a model with multiple layer types
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.bn = nn.BatchNorm1d(120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            out = self.conv1(x)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.avg_pool2d(out, 2)
            out = out.view(out.shape[0], -1)
            out = F.relu(self.fc1(out))
            out = self.bn(out)
            out = F.relu(self.fc2(out))
            out = self.fc3(out)
            return out

    model_net = Net()
    model_net.send(bob)

    ## Forward on the remote model
    pred = model_net(tensor_remote)

    assert pred.is_wrapper
    assert isinstance(pred.child, syft.PointerTensor)

    model_net.get()

    for p in model_net.parameters():
        assert isinstance(p, torch.nn.Parameter)
        assert not hasattr(p, "child")


def test_encrypt_decrypt(workers):
    bob, alice, james = (workers["bob"], workers["alice"], workers["james"])

    x = torch.randint(10, (1, 5), dtype=torch.float32)
    x_encrypted = x.encrypt(workers=[bob, alice], crypto_provider=james, base=10)
    x_decrypted = x_encrypted.decrypt()
    assert torch.all(torch.eq(x_decrypted, x))

    x = torch.randint(10, (1, 5), dtype=torch.float32)
    x_encrypted = x.encrypt(workers=[bob, alice], crypto_provider=james)
    x_decrypted = x_encrypted.decrypt()
    assert torch.all(torch.eq(x_decrypted, x))

    x = torch.randint(10, (1, 5), dtype=torch.float32)
    public, private = syft.frameworks.torch.he.paillier.keygen()
    x_encrypted = x.encrypt(protocol="paillier", public_key=public)
    x_decrypted = x_encrypted.decrypt(protocol="paillier", private_key=private)
    assert torch.all(torch.eq(x_decrypted, x))


def test_get_response():
    test_func = lambda x: x
    t = torch.tensor(73)
    # a non overloaded function
    setattr(torch, "_test_func", test_func)
    result = torch.Tensor._get_response("torch._test_func", t, {})
    delattr(torch, "_test_func")
    assert t == result
