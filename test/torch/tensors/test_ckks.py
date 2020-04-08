import pytest
import syft as sy
import torch as th
import syft.frameworks.tenseal as ts


@pytest.fixture(scope="module")
def context():
    return ts.context(
        ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60]
    )


@pytest.mark.parametrize(
    "t",
    [
        # vector
        th.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        # matrix
        th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        # 3D-tensor
        th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
    ],
)
def test_enc_dec(context, t):
    t_encrypted = t.encrypt("ckks", context=context, scale=2 ** 40)
    t_decrypted = t_encrypted.decrypt("ckks")

    assert t_decrypted.shape == t.shape
    # ckks might introduce some error
    diff = th.abs(t - t_decrypted)
    assert not th.any(diff > 0.1)


@pytest.mark.parametrize(
    "t1, t2",
    [
        # vector
        (th.tensor([1, 2, 3, 4, 5, 6, 7, 8]), th.tensor([1, 2, 3, 4, 5, 6, 7, 8])),
        # matrix
        (th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])),
        # 3D-tensor
        (
            th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
            th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
        ),
    ],
)
def test_add_encrypted(context, t1, t2):
    t1_encrypted = t1.encrypt("ckks", context=context, scale=2 ** 40)
    t2_encrypted = t2.encrypt("ckks", context=context, scale=2 ** 40)
    t_add = t1_encrypted + t2_encrypted

    assert isinstance(t_add.child, sy.CKKSTensor)
    t_decrypted = t_add.decrypt("ckks")

    # ckks might introduce some error
    diff = th.abs((t1 + t2) - t_decrypted)
    assert not th.any(diff > 0.1)


@pytest.mark.parametrize(
    "t1, t2",
    [
        # vector
        (th.tensor([1, 2, 3, 4, 5, 6, 7, 8]), th.tensor([1, 2, 3, 4, 5, 6, 7, 8])),
        # matrix
        (th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])),
        # 3D-tensor
        (
            th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
            th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
        ),
    ],
)
def test_add_plain(context, t1, t2):
    t1_encrypted = t1.encrypt("ckks", context=context, scale=2 ** 40)
    t_add = t1_encrypted + t2

    assert isinstance(t_add.child, sy.CKKSTensor)
    t_decrypted = t_add.decrypt("ckks")

    # ckks might introduce some error
    diff = th.abs((t1 + t2) - t_decrypted)
    assert not th.any(diff > 0.1)


@pytest.mark.parametrize(
    "t1, t2",
    [
        # vector
        (th.tensor([1, 2, 3, 4, 5, 6, 7, 8]), th.tensor([1, 2, 3, 4, 5, 6, 7, 8])),
        # matrix
        (th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])),
        # 3D-tensor
        (
            th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
            th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
        ),
    ],
)
def test_sub_encrypted(context, t1, t2):
    t1_encrypted = t1.encrypt("ckks", context=context, scale=2 ** 40)
    t2_encrypted = t2.encrypt("ckks", context=context, scale=2 ** 40)
    t_add = t1_encrypted - t2_encrypted

    assert isinstance(t_add.child, sy.CKKSTensor)
    t_decrypted = t_add.decrypt("ckks")

    # ckks might introduce some error
    diff = th.abs((t1 - t2) - t_decrypted)
    assert not th.any(diff > 0.1)


@pytest.mark.parametrize(
    "t1, t2",
    [
        # vector
        (th.tensor([1, 2, 3, 4, 5, 6, 7, 8]), th.tensor([1, 2, 3, 4, 5, 6, 7, 8])),
        # matrix
        (th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])),
        # 3D-tensor
        (
            th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
            th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
        ),
    ],
)
def test_sub_plain(context, t1, t2):
    t1_encrypted = t1.encrypt("ckks", context=context, scale=2 ** 40)
    t_add = t1_encrypted - t2

    assert isinstance(t_add.child, sy.CKKSTensor)
    t_decrypted = t_add.decrypt("ckks")

    # ckks might introduce some error
    diff = th.abs((t1 - t2) - t_decrypted)
    assert not th.any(diff > 0.1)


@pytest.mark.parametrize(
    "t1, t2",
    [
        # vector
        (th.tensor([1, 2, 3, 4, 5, 6, 7, 8]), th.tensor([1, 2, 3, 4, 5, 6, 7, 8])),
        # matrix
        (th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])),
        # 3D-tensor
        (
            th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
            th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
        ),
    ],
)
def test_mul_encrypted(context, t1, t2):
    t1_encrypted = t1.encrypt("ckks", context=context, scale=2 ** 40)
    t2_encrypted = t2.encrypt("ckks", context=context, scale=2 ** 40)
    t_add = t1_encrypted * t2_encrypted

    assert isinstance(t_add.child, sy.CKKSTensor)
    t_decrypted = t_add.decrypt("ckks")

    # ckks might introduce some error
    diff = th.abs((t1 * t2) - t_decrypted)
    assert not th.any(diff > 0.1)


@pytest.mark.parametrize(
    "t1, t2",
    [
        # vector
        (th.tensor([1, 2, 3, 4, 5, 6, 7, 8]), th.tensor([1, 2, 3, 4, 5, 6, 7, 8])),
        # matrix
        (th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), th.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])),
        # 3D-tensor
        (
            th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
            th.tensor([[[9, 6], [1, 6]], [[1, 1], [1, 3]]]),
        ),
    ],
)
def test_mul_plain(context, t1, t2):
    t1_encrypted = t1.encrypt("ckks", context=context, scale=2 ** 40)
    t_add = t1_encrypted * t2

    assert isinstance(t_add.child, sy.CKKSTensor)
    t_decrypted = t_add.decrypt("ckks")

    # ckks might introduce some error
    diff = th.abs((t1 * t2) - t_decrypted)
    assert not th.any(diff > 0.1)
