import syft as sy
import torch as th


def test_parameter_serde():
    param = th.nn.parameter.Parameter(th.tensor([1.0, 2, 3]), requires_grad=True)
    # Setting grad manually to check it is passed through serialization
    param.grad = th.randn_like(param)

    blob = param.serialize()

    param2 = sy.deserialize(blob=blob)

    assert (param == param2).all()
    assert (param2.grad == param2.grad).all()
    assert param2.requires_grad == param2.requires_grad


def test_linear_parameters_serde():
    # Problem: torch.nn.modules.linear imports Parameter constructor before syft replaces it
    # I.e. th.nn.parameter.Parameter is replaced while th.nn.modules.linear.Parameter is original
    linear = th.nn.Linear(5, 1)
    param = linear.weight

    # As a result, this fails:
    assert hasattr(param, "id")

    blob = param.serialize()

    param2 = sy.deserialize(blob=blob)

    assert (param == param2).all()
    assert (param2.grad == param2.grad).all()
    assert param2.requires_grad == param2.requires_grad


def test_linear_grad_serde():
    # Let's get through problem above with this silly code
    th.nn.modules.linear.Parameter = th.nn.parameter.Parameter

    # Now param is created with syft Parameter wrapper
    linear = th.nn.Linear(5, 1)
    param = linear.weight
    assert hasattr(param, "id")

    # Induce grads on linear weights
    out = linear(th.randn(5, 5))
    out.backward()
    assert param.grad is not None

    blob = param.serialize()

    param2 = sy.deserialize(blob=blob)

    assert (param == param2).all()
    assert (param2.grad == param2.grad).all()
    assert param2.requires_grad == param2.requires_grad
