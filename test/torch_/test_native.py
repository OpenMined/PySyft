import syft.torch_ as th
import torch as _th


def test_native_tensor_constructor():
    x = th.tensor([1, 2, 3, 4])

    assert isinstance(x, _th.Tensor)
    assert isinstance(x, th.Tensor.type)
    assert isinstance(x, th.tensor.type)

    assert (x == th.Tensor([1, 2, 3, 4])).all()


def test_native_parameter_constructor():
    x = th.tensor([1, 2, 3, 4.])

    assert isinstance(x, _th.Tensor)
    assert isinstance(x, th.Tensor.type)
    assert isinstance(x, th.tensor.type)

    p = th.Parameter(x)

    assert isinstance(p, th.Parameter.type)
    assert (p.data == th.Tensor([1, 2, 3, 4])).all()