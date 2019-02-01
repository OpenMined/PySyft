"""Tests relative to verifying the hook process behaves properly."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import syft
from syft.exceptions import RemoteTensorFoundError
from syft.frameworks.torch.tensors import PointerTensor
from test.conftest import hook


def test__init__(hook):
    assert torch.torch_hooked
    assert hook.torch.__version__ == torch.__version__


def test_torch_attributes():
    with pytest.raises(RuntimeError):
        syft.torch._command_guard("false_command", "torch_modules")

    assert syft.torch._is_command_valid_guard("torch.add", "torch_modules")
    assert not syft.torch._is_command_valid_guard("false_command", "torch_modules")

    syft.torch._command_guard("torch.add", "torch_modules", get_native=False)


def test_worker_registration(hook):
    me = hook.local_worker
    boris = syft.VirtualWorker(id="boris", hook=hook, is_client_worker=False)
    me.add_workers([boris])

    worker = me.get_worker(boris)

    assert boris == worker


def test_pointer_found_exception(hook):
    me = hook.local_worker

    instance_id = str(int(10e10 * random.random()))
    ptr_id = int(10e10 * random.random())

    alice = syft.VirtualWorker(id=f"alice{instance_id}", hook=hook, is_client_worker=False)

    pointer = PointerTensor(id=ptr_id, location=alice, owner=me)
    try:
        raise RemoteTensorFoundError(pointer)
    except RemoteTensorFoundError as err:
        err_pointer = err.pointer
        assert isinstance(err_pointer, PointerTensor)
        assert err_pointer.id == ptr_id


def test_build_get_child_type():
    from syft.frameworks.torch.hook_args import build_rule, build_get_tensor_type

    x = torch.Tensor([1, 2, 3])
    args = (x, [[1, x]])
    rule = build_rule(args)

    get_child_type_function = build_get_tensor_type(rule)

    tensor_type = get_child_type_function(args)
    assert tensor_type == torch.Tensor


@pytest.mark.parametrize("attr", ["abs"])
def test_get_pointer_unary_method(attr, hook):
    instance_id = str(int(10e10 * random.random()))

    bob = syft.VirtualWorker(id=f"bob{instance_id}", hook=hook, is_client_worker=False)
    x = torch.Tensor([1, 2, 3])
    native_method = getattr(x, f"native_{attr}")
    expected = native_method()
    x_ptr = x.send(bob)
    method = getattr(x_ptr, attr)
    res_ptr = method()
    res = res_ptr.get()
    assert (res == expected).all()


@pytest.mark.parametrize("attr", ["add", "mul"])
def test_get_pointer_binary_method(attr, hook):
    instance_id = str(int(10e10 * random.random()))

    bob = syft.VirtualWorker(id=f"bob{instance_id}", hook=hook, is_client_worker=False)

    x = torch.Tensor([1, 2, 3])
    native_method = getattr(x, f"native_{attr}")
    expected = native_method(x)
    x_ptr = x.send(bob)
    method = getattr(x_ptr, attr)
    res_ptr = method(x_ptr)
    res = res_ptr.get()
    assert (res == expected).all()


@pytest.mark.parametrize("attr", ["abs"])
def test_get_pointer_to_pointer_unary_method(attr, hook):
    instance_id = str(int(10e10 * random.random()))

    bob = syft.VirtualWorker(id=f"bob{instance_id}", hook=hook, is_client_worker=False)
    alice = syft.VirtualWorker(id=f"alice{instance_id}", hook=hook, is_client_worker=False)

    x = torch.Tensor([1, 2, 3])
    native_method = getattr(x, f"native_{attr}")
    expected = native_method()
    x_ptr = x.send(bob).send(alice)
    method = getattr(x_ptr, attr)
    res_ptr = method()
    res = res_ptr.get().get()
    assert (res == expected).all()


@pytest.mark.parametrize("attr", ["add", "mul"])
def test_get_pointer_to_pointer_binary_method(attr, hook):
    instance_id = str(int(10e10 * random.random()))

    bob = syft.VirtualWorker(id=f"bob{instance_id}", hook=hook, is_client_worker=False)
    alice = syft.VirtualWorker(id=f"alice{instance_id}", hook=hook, is_client_worker=False)

    x = torch.Tensor([1, 2, 3])
    native_method = getattr(x, f"native_{attr}")
    expected = native_method(x)
    x_ptr = x.send(bob).send(alice)
    method = getattr(x_ptr, attr)
    res_ptr = method(x_ptr)
    res = res_ptr.get().get()
    assert (res == expected).all()


@pytest.mark.parametrize("attr", ["relu", "celu", "elu"])
def test_hook_module_functional(attr, hook):
    instance_id = str(int(10e10 * random.random()))

    bob = syft.VirtualWorker(id=f"bob{instance_id}", hook=hook, is_client_worker=False)

    attr = getattr(F, attr)
    x = torch.Tensor([1, -1, 3, 4])
    expected = attr(x)
    x_ptr = x.send(bob)
    res_ptr = attr(x_ptr)
    res = res_ptr.get()
    assert (res == expected).all()


@pytest.mark.parametrize("attr", ["relu", "celu", "elu"])
def test_functional_same_in_both_imports(attr):
    """This function tests that the hook modifies the behavior of
    torch.nn.function regardless of the import namespace
    """
    fattr = getattr(F, attr)
    tattr = getattr(torch.nn.functional, attr)
    x = torch.Tensor([1, -1, 3, 4])
    assert (fattr(x) == tattr(x)).all()


def test_hook_tensor(hook):
    instance_id = str(int(10e10 * random.random()))

    bob = syft.VirtualWorker(id=f"bob{instance_id}", hook=hook, is_client_worker=False)

    x = torch.tensor([1.0, -1.0, 3.0, 4.0], requires_grad=True)
    x.send(bob)
    x = torch.tensor([1.0, -1.0, 3.0, 4.0], requires_grad=True)[0:2]
    x.send(bob)


def test_properties():
    x = torch.Tensor([1, -1, 3, 4])
    assert x.is_wrapper is False


def test_signature_cache_change():
    """Tests that calls to the same method using a different
    signature works correctly. We cache signatures in the
    hook.build_hook_args_function dictionary but sometimes they
    are incorrect if we use the same method with different
    parameter types. So, we need to test to make sure that
    this cache missing fails gracefully. This test tests
    that for the .div(tensor) .div(int) method."""

    x = torch.Tensor([1, 2, 3])
    y = torch.Tensor([1, 2, 3])

    z = x.div(y)
    z = x.div(2)
    z = x.div(y)

    assert True


def test_parameter_hooking():
    """Test custom nn.Module and parameter auto listing in m.parameters()"""

    class MyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.some_params = torch.nn.Parameter(torch.tensor([5.0]))

    m = MyLayer()
    out = list(m.parameters())
    assert len(out) == 1
    assert out[0] == m.some_params


def test_torch_module_hook(hook):
    """Tests sending and getting back torch nn module like nn.Linear"""

    instance_id = str(int(10e10 * random.random()))

    bob = syft.VirtualWorker(id=f"bob{instance_id}", hook=hook, is_client_worker=False)

    model = nn.Linear(2, 1)
    model_ptr = model.send(bob)
    res = model_ptr.get()
