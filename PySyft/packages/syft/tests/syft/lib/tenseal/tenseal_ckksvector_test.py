# stdlib
from typing import Any
from typing import Sequence

# third party
import pytest

# syft absolute
import syft as sy

# syft relative
from .utils_test import decrypt

ts = pytest.importorskip("tenseal")
sy.load("tenseal")


def _almost_equal(vec1: Sequence, vec2: Sequence, precision_pow_ten: int = 1) -> None:
    upper_bound = pow(10, -precision_pow_ten)
    assert pytest.approx(vec1, abs=upper_bound) == vec2


@pytest.fixture(scope="function")
def context() -> Any:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=1
    )
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()
    return context


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_sanity(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    enc_v1 = ts.ckks_vector(context, v1)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)
    enc_v1_ptr.link_context(ctx_ptr)

    result = decrypt(context, enc_v1_ptr)
    _almost_equal(result, [0, 1, 2, 3, 4])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_add(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 + v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_vector(context, v1)
    enc_v2 = ts.ckks_vector(context, v2)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)
    enc_v2_ptr = enc_v2.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    # add
    result_enc_ptr = enc_v1_ptr + enc_v2_ptr

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, expected)

    # add inplace
    enc_v1_ptr += enc_v2_ptr

    result = decrypt(context, enc_v1_ptr)
    _almost_equal(result, expected)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_sub(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 - v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_vector(context, v1)
    enc_v2 = ts.ckks_vector(context, v2)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)
    enc_v2_ptr = enc_v2.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    # sub
    result_enc_ptr = enc_v1_ptr - enc_v2_ptr

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, expected)

    # sub inplace
    enc_v1_ptr -= enc_v2_ptr

    result = decrypt(context, enc_v1_ptr)
    _almost_equal(result, expected)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_mul(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 * v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_vector(context, v1)
    enc_v2 = ts.ckks_vector(context, v2)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)
    enc_v2_ptr = enc_v2.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    # mul
    result_enc_ptr = enc_v1_ptr * enc_v2_ptr

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, expected)

    # mul inplace
    enc_v1_ptr *= enc_v2_ptr

    result = decrypt(context, enc_v1_ptr)
    _almost_equal(result, expected)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_iadd(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 + v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_vector(context, v1)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    # iadd
    result_enc_ptr = enc_v1_ptr + v2

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, expected)

    # radd
    result_enc_ptr = v2 + enc_v1_ptr

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, expected)

    # iadd inplace
    enc_v1_ptr += v2

    result = decrypt(context, enc_v1_ptr)
    _almost_equal(result, expected)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_isub(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 - v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_vector(context, v1)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    # isub
    result_enc_ptr = enc_v1_ptr - v2

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, expected)

    # rsub
    result_enc_ptr = v2 - enc_v1_ptr

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, [v2 - v1 for v1, v2 in zip(v1, v2)])


@pytest.mark.vendor(lib="tenseal")
def ptest_tenseal_ckksvector_imul(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 * v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_vector(context, v1)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    # imul
    result_enc_ptr = enc_v1_ptr * v2

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, expected)

    # rmul
    result_enc_ptr = v2 * enc_v1_ptr

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, expected)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_power(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    enc_v1 = ts.ckks_vector(context, [0, 1, 2, 3, 4])

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result_enc_ptr = enc_v1_ptr ** 3

    result = decrypt(context, result_enc_ptr)

    _almost_equal(result, [0, 1, 8, 27, 64])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_negation(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    enc_v1 = ts.ckks_vector(context, [1, 2, 3, 4, 5])

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result_enc_ptr = -enc_v1_ptr

    result = decrypt(context, result_enc_ptr)

    _almost_equal(result, [-1, -2, -3, -4, -5])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_square(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    enc_v1 = ts.ckks_vector(context, [0, 1, 2, 3, 4])

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result_enc_ptr = enc_v1_ptr.square()

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, [0, 1, 4, 9, 16])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_sum(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    enc_v1 = ts.ckks_vector(context, [0, 1, 2, 3, 4])

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result_enc_ptr = enc_v1_ptr.sum()

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, [10])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_polyval(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    polynom = [1, 2, 3, 4]
    enc_v1 = ts.ckks_vector(context, [-2, 2])

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result_enc_ptr = enc_v1_ptr.polyval(polynom)

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, [-23, 49])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_dot(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]

    enc_v1 = ts.ckks_vector(context, v1)
    enc_v2 = ts.ckks_vector(context, v2)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)
    enc_v2_ptr = enc_v2.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    result_enc_ptr2 = enc_v1_ptr.dot(enc_v2_ptr)

    result = decrypt(context, result_enc_ptr2)
    _almost_equal(result, [10])

    # inplace

    enc_v1_ptr.dot_(enc_v2_ptr)
    result = decrypt(context, enc_v1_ptr)
    _almost_equal(result, [10])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckksvector_matmul(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    enc_v1 = ts.ckks_vector(context, v1)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    matrix = [
        [73, 0.5, 8],
        [81, -5, 66],
        [-100, -78, -2],
        [0, 9, 17],
        [69, 11, 10],
    ]

    # matmul
    result_enc_ptr = enc_v1_ptr.matmul(matrix)

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, [157, -90, 153])

    # mm
    result_enc_ptr = enc_v1_ptr.mm(matrix)

    result = decrypt(context, result_enc_ptr)
    _almost_equal(result, [157, -90, 153])

    # inplace
    enc_v1_ptr.mm_(matrix)

    result = decrypt(context, enc_v1_ptr)
    _almost_equal(result, [157, -90, 153])
