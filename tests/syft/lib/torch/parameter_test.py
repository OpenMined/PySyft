import syft as sy
import torch as th


def test_parameter_serde():

    linear = th.nn.Linear(5, 1)
    linear(th.randn(5, 5)).backward()

    w = linear.weight
    blob = w.serialize()

    w2 = sy.deserialize(blob=blob)

    assert (w == w2).all()
    assert (w.grad == w2.grad).all()
    assert w.requires_grad == w2.requires_grad
