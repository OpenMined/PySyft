import pytest
import torch as th
import numpy as np

from syft.frameworks.torch.mpc.fss import DPF, DIF, n


@pytest.mark.parametrize("op", ["eq", "le"])
def test_fss_class(op):
    class_ = {"eq": DPF, "le": DIF}[op]
    th_op = {"eq": np.equal, "le": np.less_equal}[op]
    gather_op = {"eq": "__add__", "le": "__add__"}[op]

    # single value
    primitive = class_.keygen(n_values=1)
    alpha, s_00, s_01, *CW = primitive
    mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)  # IID in int32
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = np.array([0])
    x_masked = x + k0[0] + k1[0]
    y0 = class_.eval(0, x_masked, *k0[1:])
    y1 = class_.eval(1, x_masked, *k1[1:])

    assert (getattr(y0, gather_op)(y1) == th_op(x, 0)).all()

    # 1D tensor
    primitive = class_.keygen(n_values=3)
    alpha, s_00, s_01, *CW = primitive
    mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = np.array([0, 2, -2])
    x_masked = x + k0[0] + k1[0]
    y0 = class_.eval(0, x_masked, *k0[1:])
    y1 = class_.eval(1, x_masked, *k1[1:])

    assert (getattr(y0, gather_op)(y1) == th_op(x, 0)).all()

    # 2D tensor
    primitive = class_.keygen(n_values=4)
    alpha, s_00, s_01, *CW = primitive
    mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = np.array([[0, 2], [-2, 0]])
    x_masked = x + k0[0].reshape(x.shape) + k1[0].reshape(x.shape)
    y0 = class_.eval(0, x_masked, *k0[1:])
    y1 = class_.eval(1, x_masked, *k1[1:])

    assert (getattr(y0, gather_op)(y1) == th_op(x, 0)).all()

    # 3D tensor
    primitive = class_.keygen(n_values=8)
    alpha, s_00, s_01, *CW = primitive
    mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = np.array([[[0, 2], [-2, 0]], [[0, 2], [-2, 0]]])
    x_masked = x + k0[0].reshape(x.shape) + k1[0].reshape(x.shape)
    y0 = class_.eval(0, x_masked, *k0[1:])
    y1 = class_.eval(1, x_masked, *k1[1:])

    assert (getattr(y0, gather_op)(y1) == th_op(x, 0)).all()


@pytest.mark.parametrize("op", ["eq", "le"])
def test_torch_to_numpy(op):
    class_ = {"eq": DPF, "le": DIF}[op]
    th_op = {"eq": th.eq, "le": th.le}[op]
    gather_op = {"eq": "__add__", "le": "__add__"}[op]

    # 1D tensor
    primitive = class_.keygen(n_values=3)
    alpha, s_00, s_01, *CW = primitive
    mask = np.random.randint(0, 2 ** n, alpha.shape, dtype=alpha.dtype)
    k0, k1 = [((alpha - mask) % 2 ** n, s_00, *CW), (mask, s_01, *CW)]

    x = th.IntTensor([0, 2, -2])
    np_x = x.numpy()
    x_masked = np_x + k0[0] + k1[0]
    y0 = class_.eval(0, x_masked, *k0[1:])
    y1 = class_.eval(1, x_masked, *k1[1:])

    np_result = getattr(y0, gather_op)(y1)
    th_result = th.native_tensor(np_result)

    assert (th_result == th_op(x, 0)).all()


@pytest.mark.parametrize("op", ["eq", "le"])
def test_using_crypto_store(workers, op):
    alice, bob, me = workers["alice"], workers["bob"], workers["me"]
    class_ = {"eq": DPF, "le": DIF}[op]
    th_op = {"eq": th.eq, "le": th.le}[op]
    gather_op = {"eq": "__add__", "le": "__add__"}[op]
    primitive = {"eq": "fss_eq", "le": "fss_comp"}[op]

    me.crypto_store.provide_primitives(primitive, [alice, bob], n_instances=6)
    k0 = alice.crypto_store.get_keys(primitive, 3, remove=True)
    k1 = bob.crypto_store.get_keys(primitive, 3, remove=True)

    x = th.IntTensor([0, 2, -2])
    np_x = x.numpy()
    x_masked = np_x + k0[0] + k1[0]
    y0 = class_.eval(0, x_masked, *k0[1:])
    y1 = class_.eval(1, x_masked, *k1[1:])

    np_result = getattr(y0, gather_op)(y1)
    th_result = th.native_tensor(np_result)

    assert (th_result == th_op(x, 0)).all()
