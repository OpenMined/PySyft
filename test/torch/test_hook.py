"""Tests relative to verifying the hook process behaves properly."""
import re

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import syft
from syft.exceptions import RemoteObjectFoundError
from syft.generic.pointers.pointer_tensor import PointerTensor


def test___init__(hook):
    assert torch.torch_hooked
    assert hook.torch.__version__ == torch.__version__


def test_torch_inplace_method():
    positives = ["__iadd__", "__imul__", "__idiv__", "share_", "get_", "encrypt_"]
    negatives = [
        "__add__",
        "__init__",
        "__str__",
        "share",
        "get",
        "encrypt",
        "__foo",
        "_bar",
        "baz__",
    ]
    for pos in positives:
        assert re.search(syft.framework._inplace_pattern, pos)
    for neg in negatives:
        assert not re.search(syft.framework._inplace_pattern, neg)


def test_torch_attributes():
    with pytest.raises(RuntimeError):
        syft.framework._command_guard("false_command")

    assert syft.framework._is_command_valid_guard("torch.add")
    assert not syft.framework._is_command_valid_guard("false_command")

    syft.framework._command_guard("torch.add", get_native=False)


def test_worker_registration(hook, workers):
    boris = syft.VirtualWorker(id="boris", hook=hook, is_client_worker=False)

    workers["me"].add_workers([boris])
    worker = workers["me"].get_worker(boris)

    assert boris == worker


def test_pointer_found_exception(workers):
    ptr_id = syft.ID_PROVIDER.pop()
    pointer = PointerTensor(id=ptr_id, location=workers["alice"], owner=workers["me"])

    try:
        raise RemoteObjectFoundError(pointer)
    except RemoteObjectFoundError as err:
        err_pointer = err.pointer
        assert isinstance(err_pointer, PointerTensor)
        assert err_pointer.id == ptr_id


def test_build_get_child_type():
    from syft.generic.frameworks.hook.hook_args import build_rule
    from syft.generic.frameworks.hook.hook_args import build_get_tensor_type

    x = torch.Tensor([1, 2, 3])
    args = (x, [[1, x]])
    rule = build_rule(args)

    get_child_type_function = build_get_tensor_type(rule)
    tensor_type = get_child_type_function(args)

    assert tensor_type == torch.Tensor


@pytest.mark.parametrize("attr", ["abs"])
def test_get_pointer_unary_method(attr, workers):
    x = torch.Tensor([1, 2, 3])
    native_method = getattr(x, f"native_{attr}")
    expected = native_method()

    x_ptr = x.send(workers["bob"])
    method = getattr(x_ptr, attr)
    res_ptr = method()
    res = res_ptr.get()

    assert (res == expected).all()


@pytest.mark.parametrize("attr", ["add", "mul"])
def test_get_pointer_binary_method(attr, workers):
    x = torch.Tensor([1, 2, 3])
    native_method = getattr(x, f"native_{attr}")
    expected = native_method(x)

    x_ptr = x.send(workers["bob"])
    method = getattr(x_ptr, attr)
    res_ptr = method(x_ptr)
    res = res_ptr.get()

    assert (res == expected).all()


@pytest.mark.parametrize("attr", ["abs"])
def test_get_pointer_to_pointer_unary_method(attr, workers):
    x = torch.Tensor([1, 2, 3])
    native_method = getattr(x, f"native_{attr}")
    expected = native_method()

    x_ptr = x.send(workers["bob"]).send(workers["alice"])
    method = getattr(x_ptr, attr)
    res_ptr = method()
    res = res_ptr.get().get()

    assert (res == expected).all()


@pytest.mark.parametrize("attr", ["add", "mul"])
def test_get_pointer_to_pointer_binary_method(attr, workers):
    x = torch.Tensor([1, 2, 3])
    native_method = getattr(x, f"native_{attr}")
    expected = native_method(x)

    x_ptr = x.send(workers["bob"]).send(workers["alice"])
    method = getattr(x_ptr, attr)
    res_ptr = method(x_ptr)
    res = res_ptr.get().get()

    assert (res == expected).all()


@pytest.mark.parametrize("attr", ["relu", "celu", "elu"])
def test_hook_module_functional(attr, workers):
    attr = getattr(F, attr)
    x = torch.Tensor([1, -1, 3, 4])
    expected = attr(x)

    x_ptr = x.send(workers["bob"])
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


def test_hook_tensor(workers):
    x = torch.tensor([1.0, -1.0, 3.0, 4.0], requires_grad=True)
    x.send(workers["bob"])
    x = torch.tensor([1.0, -1.0, 3.0, 4.0], requires_grad=True)[0:2]
    x_ptr = x.send(workers["bob"])
    assert hasattr(x_ptr, "child")


def test_properties():
    x = torch.Tensor([1, -1, 3, 4])
    assert x.is_wrapper is False


def test_signature_cache_change():
    """Tests that calls to the same method using a different
    signature works correctly. We cache signatures in the
    hook.build_unwrap_args_from_function dictionary but sometimes they
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


def test_torch_module_hook(workers):
    """Tests sending and getting back torch nn module like nn.Linear"""
    model = nn.Linear(2, 1)
    model_ptr = model.send(workers["bob"])
    model_back = model_ptr.get()

    bias = model_back.bias
    model_back.fix_precision()
    model_back.float_precision()
    assert (bias == model_back.bias).all()


def test_functional_hook():
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.einsum("ij,jk->ik", x, x)
    assert (y == torch.tensor([[7, 10], [15, 22]])).all()


def test_hook_args_and_cmd_signature_malleability():
    """Challenge the hook_arg module with methods used with different signatures"""
    a = syft.LoggingTensor().on(torch.tensor([1.0, 2]))
    b = syft.LoggingTensor().on(torch.tensor([1.0, 2]))

    r1 = a + b
    assert (r1 == syft.LoggingTensor().on(torch.tensor([2.0, 4]))).all()

    r2 = a + 1
    assert (r2 == syft.LoggingTensor().on(torch.tensor([2.0, 3]))).all()

    r3 = a + b
    assert (r3 == syft.LoggingTensor().on(torch.tensor([2.0, 4]))).all()


def test_torch_func_signature_without_tensor():
    """The hook on the args of torch commands should work even if the args
    don't contain any tensor"""
    x = torch.as_tensor((0.1307,), dtype=torch.float32, device="cpu")
    assert (x == torch.tensor([0.1307])).all()


def test_RNN_grad_set_backpropagation(workers):
    """Perform backpropagation at a remote worker and check if the gradient updates
    and properly computed within the model"""

    alice = workers["alice"]

    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
            self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
            self.i2o = nn.Linear(input_size + hidden_size, output_size)
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, input, hidden):
            combined = torch.cat((input, hidden), 1)
            hidden = self.i2h(combined)
            output = self.i2o(combined)
            output = self.softmax(output)
            return output, hidden

        def initHidden(self):
            return torch.zeros(1, self.hidden_size)

    # let's initialize a simple RNN
    n_hidden = 128
    n_letters = 57
    n_categories = 18

    rnn = RNN(n_letters, n_hidden, n_categories)

    # Let's send the model to alice, who will be responsible for the tiny computation
    alice_model = rnn.copy().send(alice)

    # Simple input for the Recurrent Neural Network
    input_tensor = torch.zeros(size=(1, 57))
    # Just set a random category for it
    input_tensor[0][20] = 1
    alice_input = input_tensor.copy().send(alice)

    label_tensor = torch.randint(low=0, high=(n_categories - 1), size=(1,))
    alice_label = label_tensor.send(alice)

    hidden_layer = alice_model.initHidden()
    alice_hidden_layer = hidden_layer.send(alice)
    # Forward pass into the NN and its hidden layers, notice how it goes sequentially
    output, alice_hidden_layer = alice_model(alice_input, alice_hidden_layer)
    criterion = nn.NLLLoss()
    loss = criterion(output, alice_label)
    # time to backpropagate...
    loss.backward()

    # now let's get the model and check if its parameters are indeed there
    model_got = alice_model.get()

    learning_rate = 0.005

    # If the gradients are there, then the backpropagation did indeed complete successfully
    for param in model_got.parameters():
        # param.grad.data would raise an exception in case it is none,
        # so we better check it beforehand
        assert param.grad.data is not None
        param.data.add_(-learning_rate, param.grad.data)


def test_local_remote_gradient_clipping(workers):
    """
    Real test case of gradient clipping for the remote and
    local parameters of an RNN
    """
    alice = workers["alice"]

    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
            self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
            self.i2o = nn.Linear(input_size + hidden_size, output_size)
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, input, hidden):
            combined = torch.cat((input, hidden), 1)
            hidden = self.i2h(combined)
            output = self.i2o(combined)
            output = self.softmax(output)
            return output, hidden

        def initHidden(self):
            return torch.zeros(1, self.hidden_size)

    # let's initialize a simple RNN
    n_hidden = 128
    n_letters = 57
    n_categories = 18

    rnn = RNN(n_letters, n_hidden, n_categories)

    # Let's send the model to alice, who will be responsible for the tiny computation
    alice_model = rnn.copy().send(alice)

    # Simple input for the Recurrent Neural Network
    input_tensor = torch.zeros(size=(1, 57))
    # Just set a random category for it
    input_tensor[0][20] = 1
    alice_input = input_tensor.copy().send(alice)

    label_tensor = torch.randint(low=0, high=(n_categories - 1), size=(1,))
    alice_label = label_tensor.send(alice)

    hidden_layer = alice_model.initHidden()
    alice_hidden_layer = hidden_layer.send(alice)
    # Forward pass into the NN and its hidden layers, notice how it goes sequentially
    output, alice_hidden_layer = alice_model(alice_input, alice_hidden_layer)
    criterion = nn.NLLLoss()
    loss = criterion(output, alice_label)
    # time to backpropagate...
    loss.backward()

    # Remote gradient clipping
    remote_parameters = alice_model.parameters()
    total_norm_remote = nn.utils.clip_grad_norm_(remote_parameters, 2)

    # Local gradient clipping
    local_alice_model = alice_model.get()
    local_parameters = local_alice_model.parameters()
    total_norm_local = nn.utils.clip_grad_norm_(local_parameters, 2)

    # Is the output of the remote gradient clipping version equal to
    # the output of the local gradient clipping version?
    assert torch.isclose(total_norm_remote.get(), total_norm_local, atol=1e-4)


# Input: None
# Output: a local PyTorch tensor with .grad attribute set to [4.5, 4.5, 4.5, 4.5]
# These operations were taken from this tutorial:
# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
def produce_tensor_with_grad():
    x = torch.ones(2, 2, requires_grad=True)
    a = torch.randn(2, 2)
    a = (a * 3) / (a - 1)
    a.requires_grad_(True)
    b = (a * a).sum()
    y = x + 2
    z = y * y * 3
    out = z.mean()
    # backward function is used for computing the gradient of the tensor 'x'
    out.backward()

    return x


def test_remote_gradient_clipping(workers):
    """
    Vanishing gradient test over
    gradients of a remote tensor
    """
    alice = workers["alice"]
    local_tensor = produce_tensor_with_grad()
    remote_tensor = local_tensor.send(alice)
    # initially, remote_tensor.grad is [4.5, 4.5, 4.5, 4.5]
    # Method operates in place
    nn.utils.clip_grad_norm_(remote_tensor, 2)
    tensor_comparison_grad = torch.Tensor([[4.5, 4.5], [4.5, 4.5]])
    tensor_comp_grad_remote = tensor_comparison_grad.send(alice)
    # Has the gradient decreased w.r.t. the initial value (i.e. was it clipped)?
    smaller_tensor_check_remote = remote_tensor.grad < tensor_comp_grad_remote

    assert 1 == smaller_tensor_check_remote.all().copy().get().item()


def test_local_gradient_clipping():
    """
    Vanishing gradient test over
    gradients of a local tensor
    """
    local_tensor = produce_tensor_with_grad()
    # initially, local_tensor.grad is [4.5, 4.5, 4.5, 4.5]
    # Method operates in place
    nn.utils.clip_grad_norm_(local_tensor, 2)
    tensor_comparison_grad = torch.Tensor([[4.5, 4.5], [4.5, 4.5]])
    # Has the gradient decreased w.r.t. the initial value (i.e. was it clipped)?
    smaller_tensor_check = local_tensor.grad < tensor_comparison_grad

    assert 1 == smaller_tensor_check.all().item()
