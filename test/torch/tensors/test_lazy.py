import torch as th


def test_lazy_add():
    x = th.tensor([[1, 2, 3]]).float()
    x = x.lazy()

    y = x + x
    assert (y.execute() == th.tensor([2, 4, 6]).float()).all()
