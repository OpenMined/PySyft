# stdlib
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy

ts = pytest.importorskip("tenseal")
sy.load_lib("tenseal")


@pytest.fixture(scope="function")
def context() -> Any:
    context = ts.context(ts.SCHEME_TYPE.BFV, 8192, 1032193)
    context.generate_galois_keys()

    return context


@pytest.fixture(scope="function")
def duet() -> Any:
    return sy.VirtualMachine().get_root_client()


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_sanity(context: Any, duet: sy.VirtualMachine) -> None:
    v1 = [0, 1, 2, 3, 4]
    enc_v1 = ts.bfv_vector(context, v1)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result = enc_v1_ptr.decrypt().get()
    assert result == [0, 1, 2, 3, 4]


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_add(context: Any, duet: sy.VirtualMachine) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 + v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.bfv_vector(context, v1)
    enc_v2 = ts.bfv_vector(context, v2)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)
    enc_v2_ptr = enc_v2.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    # add
    result_enc_ptr = enc_v1_ptr + enc_v2_ptr

    result = result_enc_ptr.decrypt().get()
    assert result == expected

    # add inplace
    enc_v1_ptr += enc_v2_ptr

    result = enc_v1_ptr.decrypt().get()
    assert result == expected


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_sub(context: Any, duet: sy.VirtualMachine) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 - v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.bfv_vector(context, v1)
    enc_v2 = ts.bfv_vector(context, v2)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)
    enc_v2_ptr = enc_v2.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    # sub
    result_enc_ptr = enc_v1_ptr - enc_v2_ptr

    result = result_enc_ptr.decrypt().get()
    assert result == expected

    # sub inplace
    enc_v1_ptr -= enc_v2_ptr

    result = enc_v1_ptr.decrypt().get()
    assert result == expected


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_mul(context: Any, duet: sy.VirtualMachine) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 * v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.bfv_vector(context, v1)
    enc_v2 = ts.bfv_vector(context, v2)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)
    enc_v2_ptr = enc_v2.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    # mul
    result_enc_ptr = enc_v1_ptr * enc_v2_ptr

    result = result_enc_ptr.decrypt().get()
    assert result == expected

    # mul inplace
    enc_v1_ptr *= enc_v2_ptr

    result = enc_v1_ptr.decrypt().get()
    assert result == expected


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_iadd(context: Any, duet: sy.VirtualMachine) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 + v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.bfv_vector(context, v1)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    # iadd
    result_enc_ptr = enc_v1_ptr + v2

    result = result_enc_ptr.decrypt().get()
    assert result == expected

    # radd
    result_enc_ptr = v2 + enc_v1_ptr

    result = result_enc_ptr.decrypt().get()
    assert result == expected

    # iadd inplace
    enc_v1_ptr += v2

    result = enc_v1_ptr.decrypt().get()
    assert result == expected


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_isub(context: Any, duet: sy.VirtualMachine) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 - v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.bfv_vector(context, v1)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    # isub
    result_enc_ptr = enc_v1_ptr - v2

    result = result_enc_ptr.decrypt().get()
    assert result == expected


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_imul(context: Any, duet: sy.VirtualMachine) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 * v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.bfv_vector(context, v1)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    # imul
    result_enc_ptr = enc_v1_ptr * v2

    result = result_enc_ptr.decrypt().get()
    assert result == expected

    # rmul
    result_enc_ptr = v2 * enc_v1_ptr

    result = result_enc_ptr.decrypt().get()
    assert result == expected
