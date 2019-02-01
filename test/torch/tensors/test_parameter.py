import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import syft
from test.conftest import hook
from test.conftest import workers

from syft.frameworks.torch.tensors import LoggingTensor


def test_param_on_pointer(workers):
    """
    """

    bob = workers["bob"]
    tensor = torch.tensor([1.0, -1.0, 3.0, 4.0])
    ptr = tensor.send(bob)
    param = Parameter(ptr)
    local_param = param.get()
    assert (local_param.data == tensor).all()


def test_param_send_get(workers):
    """
    """

    bob = workers["bob"]
    tensor = torch.tensor([1.0, -1.0, 3.0, 4.0])
    param = Parameter(data=tensor.clone())
    param_ptr = param.send(bob)
    param_back = param_ptr.get()
    assert (param_back.data == tensor).all()


def test_param_remote_binary_method(workers):
    """
    """

    bob = workers["bob"]
    tensor = torch.tensor([1.0, -1.0, 3.0, 4.0])
    param = Parameter(data=tensor.clone())
    param_ptr = param.send(bob)
    param_double_ptr = param_ptr + param_ptr
    param_double_back = param_double_ptr.get()
    double_tensor = tensor + tensor
    assert (param_double_back.data == double_tensor).all()

def test_local_param_in_nn_module_linear():
    model = nn.Linear(2, 1)
    tensor = torch.tensor([1.0, -1.0])
    res = model(tensor)

def test_remote_param_in_nn_module_linear(workers):
    bob = workers["bob"]
    model = nn.Linear(2, 1, bias=False)
    tensor = torch.tensor([1.0, -1.0])
    model_ptr = model.send(bob)
    tensor_ptr = tensor.send(bob)
    res_ptr = model_ptr(tensor_ptr)
    res = res_ptr.get()
