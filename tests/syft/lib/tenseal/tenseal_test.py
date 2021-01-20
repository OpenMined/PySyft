# third party
import pytest
import tenseal as ts

# syft absolute
import syft as sy

from typing import Sequence
from typing import Any

sy.load_lib("tenseal")


def _almost_equal(vec1: Sequence, vec2: Sequence, precision_pow_ten: int = 4) -> bool:
    if len(vec1) != len(vec2):
        return False

    upper_bound = pow(10, -precision_pow_ten)
    for v1, v2 in zip(vec1, vec2):
        if abs(v1 - v2) > upper_bound:
            return False
    return True


@pytest.fixture(scope="function")
def context() -> ts.Context:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()
    return context


@pytest.fixture(scope="function")
def duet() -> Any:
    return sy.VirtualMachine().get_root_client()


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_sanity(context: ts.Context, duet: sy.VirtualMachine) -> None:
    v1 = [0, 1, 2, 3, 4]
    enc_v1 = ts.ckks_vector(context, v1)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result = enc_v1_ptr.decrypt().get()
    assert _almost_equal(result, [0, 1, 2, 3, 4])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_add(context: ts.Context, duet: sy.VirtualMachine) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 + v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_vector(context, v1)
    enc_v2 = ts.ckks_vector(context, v2)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)
    enc_v2_ptr = enc_v2.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    result_enc_ptr = enc_v1_ptr + enc_v2_ptr

    result_dec_ptr = result_enc_ptr.decrypt()
    result = result_dec_ptr.get()
    assert _almost_equal(result, expected)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_iadd(context: ts.Context, duet: sy.VirtualMachine) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 + v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_vector(context, v1)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    # iadd
    result_enc_ptr = enc_v1_ptr + v2

    result = result_enc_ptr.decrypt().get()
    assert _almost_equal(result, expected)

    # radd
    result_enc_ptr = v2 + enc_v1_ptr

    result = result_enc_ptr.decrypt().get()
    assert _almost_equal(result, expected)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_dot(context: ts.Context, duet: sy.VirtualMachine) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]

    enc_v1 = ts.ckks_vector(context, v1)
    enc_v2 = ts.ckks_vector(context, v2)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)
    enc_v2_ptr = enc_v2.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    result_enc_ptr2 = enc_v1_ptr.dot(enc_v2_ptr)
    result_dec_ptr2 = result_enc_ptr2.decrypt()
    result2 = result_dec_ptr2.get()
    assert _almost_equal(result2, [10])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_matmul(context: ts.Context, duet: sy.VirtualMachine) -> None:
    v1 = [0, 1, 2, 3, 4]
    enc_v1 = ts.ckks_vector(context, v1)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    matrix = [
        [73, 0.5, 8],
        [81, -5, 66],
        [-100, -78, -2],
        [0, 9, 17],
        [69, 11, 10],
    ]

    result_enc_ptr3 = enc_v1_ptr.matmul(matrix)

    result_dec_ptr3 = result_enc_ptr3.decrypt()
    result3 = result_dec_ptr3.get()
    assert _almost_equal(result3, [157, -90, 153])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_power(context: ts.Context, duet: sy.VirtualMachine) -> None:
    enc_v1 = ts.ckks_vector(context, [0, 1, 2, 3, 4])

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result_enc_ptr = enc_v1_ptr ** 2

    result_dec_ptr = result_enc_ptr.decrypt()
    result = result_dec_ptr.get()

    assert _almost_equal(result, [0, 1, 4, 9, 16])
