import syft
import torch as th

from syft.frameworks.torch.mpc.fss import DPF, DIF, n


def test_DPF():
    # single value
    primitive = DPF.keygen(n_values=1)
    alpha, s_00, s_01, *CW = primitive
    mask = th.randint(0, 2 ** n, alpha.shape)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = th.tensor([0])
    x_masked = x + k0[0] + k1[0]
    y0 = DPF.eval(0, x_masked, *k0[1:])
    y1 = DPF.eval(1, x_masked, *k1[1:])

    assert ((y0 + y1) == (x == 0)).all()

    # 1D tensor
    primitive = DPF.keygen(n_values=3)
    alpha, s_00, s_01, *CW = primitive
    mask = th.randint(0, 2 ** n, alpha.shape)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = th.tensor([0, 2, -2])
    x_masked = x + k0[0] + k1[0]
    y0 = DPF.eval(0, x_masked, *k0[1:])
    y1 = DPF.eval(1, x_masked, *k1[1:])

    assert ((y0 + y1) == (x == 0)).all()

    # 2D tensor
    primitive = DPF.keygen(n_values=4)
    alpha, s_00, s_01, *CW = primitive
    mask = th.randint(0, 2 ** n, alpha.shape)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = th.tensor([[0, 2], [-2, 0]])
    x_masked = x + k0[0].reshape(x.shape) + k1[0].reshape(x.shape)
    y0 = DPF.eval(0, x_masked, *k0[1:])
    y1 = DPF.eval(1, x_masked, *k1[1:])

    assert ((y0 + y1) == (x == 0)).all()

    # 3D tensor
    primitive = DPF.keygen(n_values=8)
    alpha, s_00, s_01, *CW = primitive
    mask = th.randint(0, 2 ** n, alpha.shape)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = th.tensor([[[0, 2], [-2, 0]], [[0, 2], [-2, 0]]])
    x_masked = x + k0[0].reshape(x.shape) + k1[0].reshape(x.shape)
    y0 = DPF.eval(0, x_masked, *k0[1:])
    y1 = DPF.eval(1, x_masked, *k1[1:])

    assert ((y0 + y1) == (x == 0)).all()
