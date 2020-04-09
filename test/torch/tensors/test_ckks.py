import pytest
import syft as sy
import torch as th
import syft.frameworks.tenseal as ts


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
def test_enc_dec(t):
    context, secret_key = ts.generate_ckks_keys()
    t_encrypted = t.encrypt("ckks", context=context, scale=2 ** 40)
    t_decrypted = t_encrypted.decrypt("ckks", secret_key=secret_key)

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
def test_add_encrypted(t1, t2):
    context, secret_key = ts.generate_ckks_keys()
    t1_encrypted = t1.encrypt("ckks", context=context, scale=2 ** 40)
    t2_encrypted = t2.encrypt("ckks", context=context, scale=2 ** 40)
    t_add = t1_encrypted + t2_encrypted

    assert isinstance(t_add.child, sy.CKKSTensor)
    t_decrypted = t_add.decrypt("ckks", secret_key=secret_key)

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
def test_add_plain(t1, t2):
    context, secret_key = ts.generate_ckks_keys()
    t1_encrypted = t1.encrypt("ckks", context=context, scale=2 ** 40)
    t_add = t1_encrypted + t2

    assert isinstance(t_add.child, sy.CKKSTensor)
    t_decrypted = t_add.decrypt("ckks", secret_key=secret_key)

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
def test_sub_encrypted(t1, t2):
    context, secret_key = ts.generate_ckks_keys()
    t1_encrypted = t1.encrypt("ckks", context=context, scale=2 ** 40)
    t2_encrypted = t2.encrypt("ckks", context=context, scale=2 ** 40)
    t_add = t1_encrypted - t2_encrypted

    assert isinstance(t_add.child, sy.CKKSTensor)
    t_decrypted = t_add.decrypt("ckks", secret_key=secret_key)

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
def test_sub_plain(t1, t2):
    context, secret_key = ts.generate_ckks_keys()
    t1_encrypted = t1.encrypt("ckks", context=context, scale=2 ** 40)
    t_add = t1_encrypted - t2

    assert isinstance(t_add.child, sy.CKKSTensor)
    t_decrypted = t_add.decrypt("ckks", secret_key=secret_key)

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
def test_mul_encrypted(t1, t2):
    context, secret_key = ts.generate_ckks_keys()
    t1_encrypted = t1.encrypt("ckks", context=context, scale=2 ** 40)
    t2_encrypted = t2.encrypt("ckks", context=context, scale=2 ** 40)
    t_add = t1_encrypted * t2_encrypted

    assert isinstance(t_add.child, sy.CKKSTensor)
    t_decrypted = t_add.decrypt("ckks", secret_key=secret_key)

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
def test_mul_plain(t1, t2):
    context, secret_key = ts.generate_ckks_keys()
    t1_encrypted = t1.encrypt("ckks", context=context, scale=2 ** 40)
    t_add = t1_encrypted * t2

    assert isinstance(t_add.child, sy.CKKSTensor)
    t_decrypted = t_add.decrypt("ckks", secret_key=secret_key)

    # ckks might introduce some error
    diff = th.abs((t1 * t2) - t_decrypted)
    assert not th.any(diff > 0.1)
