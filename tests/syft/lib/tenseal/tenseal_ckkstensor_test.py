# third party
import pytest

# syft absolute
import syft as sy

from typing import Any


def _almost_equal_number(v1: Any, v2: Any, m_pow_ten: int) -> bool:
    upper_bound = pow(10, -m_pow_ten)

    return abs(v1 - v2) <= upper_bound


def _almost_equal(vec1: Any, vec2: Any, precision_pow_ten: int = 1) -> bool:
    import tenseal as ts

    if isinstance(vec1, ts.PlainTensor):
        vec1 = vec1.tolist()
    if isinstance(vec2, ts.PlainTensor):
        vec2 = vec2.tolist()

    if isinstance(vec1, (float, int)):
        return _almost_equal_number(vec1, vec2, precision_pow_ten)

    if len(vec1) != len(vec2):
        return False

    for v1, v2 in zip(vec1, vec2):
        if isinstance(v1, list):
            if not _almost_equal(v1, v2, precision_pow_ten):
                return False
        elif not _almost_equal_number(v1, v2, precision_pow_ten):
            return False
    return True


@pytest.fixture(scope="function")
def context() -> Any:
    import tenseal as ts

    sy.load_lib("tenseal")

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
def test_tenseal_ckkstensor_sanity(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    v1 = [0, 1, 2, 3, 4]
    enc_v1 = ts.ckks_tensor(context, v1)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result = enc_v1_ptr.decrypt().get()
    assert _almost_equal(result, [0, 1, 2, 3, 4])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckkstensor_add(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 + v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_tensor(context, v1)
    enc_v2 = ts.ckks_tensor(context, v2)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)
    enc_v2_ptr = enc_v2.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    # add
    result_enc_ptr = enc_v1_ptr + enc_v2_ptr

    result = result_enc_ptr.decrypt().get()
    assert _almost_equal(result, expected)

    # add inplace
    enc_v1_ptr += enc_v2_ptr

    result = enc_v1_ptr.decrypt().get()
    assert _almost_equal(result, expected)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckkstensor_sub(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 - v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_tensor(context, v1)
    enc_v2 = ts.ckks_tensor(context, v2)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)
    enc_v2_ptr = enc_v2.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    # sub
    result_enc_ptr = enc_v1_ptr - enc_v2_ptr

    result = result_enc_ptr.decrypt().get()
    assert _almost_equal(result, expected)

    # sub inplace
    enc_v1_ptr -= enc_v2_ptr

    result = enc_v1_ptr.decrypt().get()
    assert _almost_equal(result, expected)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckkstensor_mul(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 * v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_tensor(context, v1)
    enc_v2 = ts.ckks_tensor(context, v2)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)
    enc_v2_ptr = enc_v2.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    # mul
    result_enc_ptr = enc_v1_ptr * enc_v2_ptr

    result = result_enc_ptr.decrypt().get()
    assert _almost_equal(result, expected)

    # mul inplace
    enc_v1_ptr *= enc_v2_ptr

    result = enc_v1_ptr.decrypt().get()
    assert _almost_equal(result, expected)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckkstensor_iadd(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 + v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_tensor(context, v1)

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

    # iadd inplace
    enc_v1_ptr += v2

    result = enc_v1_ptr.decrypt().get()
    assert _almost_equal(result, expected)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckkstensor_isub(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 - v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_tensor(context, v1)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    # isub
    result_enc_ptr = enc_v1_ptr - v2

    result = result_enc_ptr.decrypt().get()
    assert _almost_equal(result, expected)

    # rsub
    result_enc_ptr = v2 - enc_v1_ptr

    result = result_enc_ptr.decrypt().get()
    assert _almost_equal(result, [v2 - v1 for v1, v2 in zip(v1, v2)])


@pytest.mark.vendor(lib="tenseal")
def ptest_tenseal_ckkstensor_imul(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 * v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.ckks_tensor(context, v1)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    # imul
    result_enc_ptr = enc_v1_ptr * v2

    result = result_enc_ptr.decrypt().get()
    assert _almost_equal(result, expected)

    # rmul
    result_enc_ptr = v2 * enc_v1_ptr

    result = result_enc_ptr.decrypt().get()
    assert _almost_equal(result, expected)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckkstensor_power(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    enc_v1 = ts.ckks_tensor(context, [0, 1, 2, 3, 4])

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result_enc_ptr = enc_v1_ptr ** 3

    result_dec_ptr = result_enc_ptr.decrypt()
    result = result_dec_ptr.get()

    assert _almost_equal(result, [0, 1, 8, 27, 64])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckkstensor_negation(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    enc_v1 = ts.ckks_tensor(context, [1, 2, 3, 4, 5])

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result_enc_ptr = -enc_v1_ptr

    result_dec_ptr = result_enc_ptr.decrypt()
    result = result_dec_ptr.get()

    assert _almost_equal(result, [-1, -2, -3, -4, -5])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckkstensor_square(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    enc_v1 = ts.ckks_tensor(context, [0, 1, 2, 3, 4])

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result_enc_ptr = enc_v1_ptr.square()

    result = result_enc_ptr.decrypt().get()
    assert _almost_equal(result, [0, 1, 4, 9, 16])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckkstensor_sum(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    enc_v1 = ts.ckks_tensor(context, [0, 1, 2, 3, 4])

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result_enc_ptr = enc_v1_ptr.sum()

    result = result_enc_ptr.decrypt().get()
    assert _almost_equal(result, 10)


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckkstensor_polyval(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    polynom = [1, 2, 3, 4]
    enc_v1 = ts.ckks_tensor(context, [-2, 2])

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result_enc_ptr = enc_v1_ptr.polyval(polynom)

    result = result_enc_ptr.decrypt().get()
    assert _almost_equal(result, [-23, 49])


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_ckkstensor_dot(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]

    enc_v1 = ts.ckks_tensor(context, v1)
    enc_v2 = ts.ckks_tensor(context, v2)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)
    enc_v2_ptr = enc_v2.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    result_enc_ptr2 = enc_v1_ptr.dot(enc_v2_ptr)
    result_dec_ptr2 = result_enc_ptr2.decrypt()
    result2 = result_dec_ptr2.get()
    assert _almost_equal(result2, 10)

    # inplace

    enc_v1_ptr.dot_(enc_v2_ptr)
    result_dec_ptr2 = enc_v1_ptr.decrypt()
    result2 = result_dec_ptr2.get()
    assert _almost_equal(result2, 10)
