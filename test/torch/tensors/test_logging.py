import random

import pytest
import torch
import torch.nn.functional as F
import syft

from syft.frameworks.torch.tensors.decorators import LoggingTensor


def test_wrap():
    """
    Test the .on() wrap functionality for LoggingTensor
    """

    x_tensor = torch.Tensor([1, 2, 3])
    x = LoggingTensor().on(x_tensor)

    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, LoggingTensor)
    assert isinstance(x.child.child, torch.Tensor)


def test_overwritten_method_on_log_chain():
    """
    Test method call on a chain including a log tensor
    """

    # Build a long chain tensor Wrapper>LoggingTensor>TorchTensor
    x_tensor = torch.Tensor([1, 2, 3])
    x = LoggingTensor().on(x_tensor)
    y = x.add(x)

    assert (y.child.child == x_tensor.add(x_tensor)).all()

    y = x.child.manual_add(x.child)

    assert (y.child == x_tensor.add(x_tensor)).all()


def test_method_on_log_chain():
    """
    Test method call on a chain including a log tensor
    """

    # Build a long chain tensor Wrapper>LoggingTensor>TorchTensor
    x_tensor = torch.Tensor([1, 2, 3])
    x = LoggingTensor().on(x_tensor)
    y = x.mul(x)

    assert (y.child.child == x_tensor.mul(x_tensor)).all()


@pytest.mark.parametrize("attr", ["relu", "celu", "elu"])
def test_hook_module_functional_on_log_chain(attr):
    """
    Test torch function call on a chain including a log tensor
    """

    attr = getattr(F, attr)
    x = torch.Tensor([1, -1, 3, 4])
    expected = attr(x)

    x_log = LoggingTensor().on(x)
    res_log = attr(x_log)
    res = res_log.child.child

    assert (res == expected).all()


def test_function_on_log_chain():
    """
    Test torch function call on a chain including a log tensor
    """

    x = LoggingTensor().on(torch.Tensor([1, -1, 3]))
    y = F.relu(x)

    assert (y.child.child == torch.Tensor([1, 0, 3])).all()


def test_send_get_log_chain(workers):
    """
    Test sending and getting back a chain including a logtensor
    """

    # Build a long chain tensor Wrapper>LoggingTensor>TorchTensor
    x_tensor = torch.Tensor([1, 2, 3])
    x = LoggingTensor().on(x_tensor)
    x_ptr = x.send(workers["bob"])
    x_back = x_ptr.get()

    assert (x_back.child.child == x_tensor).all()


def test_inplace_send_get_log_chain(workers):
    """
    Test sending and getting back a chain including a logtensor
    """

    # Build a long chain tensor Wrapper>LoggingTensor>TorchTensor
    x_tensor = torch.Tensor([1, 2, 3])
    x = LoggingTensor().on(x_tensor)
    x_ptr = x.send_(workers["bob"])
    x_back = x_ptr.get_()

    assert (x_back.child.child == x_tensor).all()


def test_remote_method_on_log_chain(workers):
    """
    Test remote method call on a chain including a log tensor
    """

    # Build a long chain tensor Wrapper>LoggingTensor>TorchTensor
    x_tensor = torch.Tensor([1, 2, 3])
    x = LoggingTensor().on(x_tensor)
    x_ptr = x.send(workers["bob"])
    y_ptr = F.relu(x_ptr)
    y = y_ptr.get()

    assert (y.child.child == F.relu(x_tensor)).all()


def test_remote_function_on_log_chain(workers):
    """
    Test remote function call on a chain including a log tensor
    """

    # Build a long chain tensor Wrapper>LoggingTensor>TorchTensor
    x_tensor = torch.Tensor([1, 2, 3])
    x = LoggingTensor().on(x_tensor)
    x_ptr = x.send(workers["bob"])
    y_ptr = x_ptr.add(x_ptr)
    y = y_ptr.get()

    assert (y.child.child == x_tensor.add(x_tensor)).all()


def test_print_log_chain():
    """
    Test sending and getting back a chain including a logtensor
    """

    # Build a long chain tensor Wrapper>LoggingTensor>TorchTensor
    x_tensor = torch.Tensor([1, 2, 3])
    x = LoggingTensor().on(x_tensor)

    assert isinstance(x.__str__(), str)
    assert isinstance(x.__repr__(), str)
