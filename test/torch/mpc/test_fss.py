import pytest
import torch as th

from syft.frameworks.torch.mpc.fss import DPF, DIF, n


@pytest.mark.parametrize("op", ["eq", "le"])
def test_fss_class(op):
    class_ = {"eq": DPF, "le": DIF}[op]
    th_op = {"eq": th.eq, "le": th.le}[op]
    gather_op = {"eq": "__add__", "le": "__xor__"}[op]

    # single value
    primitive = class_.keygen(n_values=1)
    alpha, s_00, s_01, *CW = primitive
    mask = th.randint(0, 2 ** n, alpha.shape)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = th.tensor([0])
    x_masked = x + k0[0] + k1[0]
    y0 = class_.eval(0, x_masked, *k0[1:])
    y1 = class_.eval(1, x_masked, *k1[1:])

    assert (getattr(y0, gather_op)(y1) == th_op(x, 0)).all()

    # 1D tensor
    primitive = class_.keygen(n_values=3)
    alpha, s_00, s_01, *CW = primitive
    mask = th.randint(0, 2 ** n, alpha.shape)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = th.tensor([0, 2, -2])
    x_masked = x + k0[0] + k1[0]
    y0 = class_.eval(0, x_masked, *k0[1:])
    y1 = class_.eval(1, x_masked, *k1[1:])

    assert (getattr(y0, gather_op)(y1) == th_op(x, 0)).all()

    # 2D tensor
    primitive = class_.keygen(n_values=4)
    alpha, s_00, s_01, *CW = primitive
    mask = th.randint(0, 2 ** n, alpha.shape)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = th.tensor([[0, 2], [-2, 0]])
    x_masked = x + k0[0].reshape(x.shape) + k1[0].reshape(x.shape)
    y0 = class_.eval(0, x_masked, *k0[1:])
    y1 = class_.eval(1, x_masked, *k1[1:])

    assert (getattr(y0, gather_op)(y1) == th_op(x, 0)).all()

    # 3D tensor
    primitive = class_.keygen(n_values=8)
    alpha, s_00, s_01, *CW = primitive
    mask = th.randint(0, 2 ** n, alpha.shape)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = th.tensor([[[0, 2], [-2, 0]], [[0, 2], [-2, 0]]])
    x_masked = x + k0[0].reshape(x.shape) + k1[0].reshape(x.shape)
    y0 = class_.eval(0, x_masked, *k0[1:])
    y1 = class_.eval(1, x_masked, *k1[1:])

    assert (getattr(y0, gather_op)(y1) == th_op(x, 0)).all()
