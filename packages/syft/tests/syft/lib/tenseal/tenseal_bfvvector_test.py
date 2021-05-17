# stdlib
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy

# syft relative
from .utils_test import decrypt

ts = pytest.importorskip("tenseal")
sy.load("tenseal")


@pytest.fixture(scope="function")
def context() -> Any:
    context = ts.context(ts.SCHEME_TYPE.BFV, 8192, 1032193, n_threads=1)
    return context


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_sanity(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    enc_v1 = ts.bfv_vector(context, v1)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result = decrypt(context, enc_v1_ptr)
    assert result == [0, 1, 2, 3, 4]


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_add(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 + v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.bfv_vector(context, v1)
    enc_v2 = ts.bfv_vector(context, v2)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)
    enc_v2_ptr = enc_v2.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    # add
    result_enc_ptr = enc_v1_ptr + enc_v2_ptr

    result = decrypt(context, result_enc_ptr)
    assert result == expected

    # add inplace
    enc_v1_ptr += enc_v2_ptr

    result = decrypt(context, enc_v1_ptr)
    assert result == expected


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_sub(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 - v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.bfv_vector(context, v1)
    enc_v2 = ts.bfv_vector(context, v2)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)
    enc_v2_ptr = enc_v2.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    # sub
    result_enc_ptr = enc_v1_ptr - enc_v2_ptr

    result = decrypt(context, result_enc_ptr)
    assert result == expected

    # sub inplace
    enc_v1_ptr -= enc_v2_ptr

    result = decrypt(context, enc_v1_ptr)
    assert result == expected


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_mul(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 * v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.bfv_vector(context, v1)
    enc_v2 = ts.bfv_vector(context, v2)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)
    enc_v2_ptr = enc_v2.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)
    enc_v2_ptr.link_context(ctx_ptr)

    # mul
    result_enc_ptr = enc_v1_ptr * enc_v2_ptr

    result = decrypt(context, result_enc_ptr)
    assert result == expected

    # mul inplace
    enc_v1_ptr *= enc_v2_ptr

    result = decrypt(context, enc_v1_ptr)
    assert result == expected


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_iadd(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 + v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.bfv_vector(context, v1)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    # iadd
    result_enc_ptr = enc_v1_ptr + v2

    result = decrypt(context, result_enc_ptr)
    assert result == expected

    # radd
    result_enc_ptr = v2 + enc_v1_ptr

    result = decrypt(context, result_enc_ptr)
    assert result == expected

    # iadd inplace
    enc_v1_ptr += v2

    result = decrypt(context, enc_v1_ptr)
    assert result == expected


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_isub(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 - v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.bfv_vector(context, v1)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    # isub
    result_enc_ptr = enc_v1_ptr - v2

    result = decrypt(context, result_enc_ptr)
    assert result == expected


@pytest.mark.vendor(lib="tenseal")
def test_tenseal_bfvvector_imul(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    v1 = [0, 1, 2, 3, 4]
    v2 = [4, 3, 2, 1, 0]
    expected = [v1 * v2 for v1, v2 in zip(v1, v2)]

    enc_v1 = ts.bfv_vector(context, v1)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    # imul
    result_enc_ptr = enc_v1_ptr * v2

    result = decrypt(context, result_enc_ptr)
    assert result == expected

    # rmul
    result_enc_ptr = v2 * enc_v1_ptr

    result = decrypt(context, result_enc_ptr)
    assert result == expected
