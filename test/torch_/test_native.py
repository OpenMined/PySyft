import syft.torch_ as th_
import torch as th
from torch import nn


def test_lowercase_tensor_constructor():
    x = th.tensor([1, 2, 3, 4])

    assert isinstance(x, th.Tensor)
    assert (x == th.Tensor([1, 2, 3, 4])).all()


def test_uppercase_tensor_constructor():
    x = th.Tensor([1, 2, 3, 4])

    assert isinstance(x, th.Tensor)
    assert (x == th.Tensor([1, 2, 3, 4])).all()


def test_parameter_constructor():
    x = th.tensor([1, 2, 3, 4.0])

    assert isinstance(x, th.Tensor)

    p = nn.Parameter(x)

    assert isinstance(p, nn.Parameter)
    assert (p.data == th.Tensor([1, 2, 3, 4])).all()
