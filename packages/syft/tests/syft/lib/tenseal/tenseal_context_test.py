# stdlib
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy

ts = pytest.importorskip("tenseal")
sy.load("tenseal")


@pytest.fixture(scope="function")
def context() -> Any:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60],
        n_threads=1,
    )
    context.global_scale = pow(2, 40)
    return context


@pytest.mark.vendor(lib="tenseal")
def test_context_send(context: Any, root_client: sy.VirtualMachineClient) -> None:
    """Test sending a TenSEAL context

    Args:
        context: TenSEAL context to test
    """
    assert len(root_client.store) == 0

    _ = context.send(root_client)

    assert len(root_client.store) == 1


@pytest.mark.parametrize("scheme", [ts.SCHEME_TYPE.CKKS, ts.SCHEME_TYPE.BFV])
@pytest.mark.vendor(lib="tenseal")
def test_scheme_send(scheme: Any, client: sy.VirtualMachineClient) -> None:
    """Test sending a TenSEAL scheme

    Args:
        scheme: TenSEAL scheme to test
    """
    st_ptr = scheme.send(client, pointable=True)
    assert st_ptr.get() == scheme


@pytest.mark.vendor(lib="tenseal")
def test_context_link(context: Any, root_client: sy.VirtualMachineClient) -> None:
    v1 = [0, 1, 2, 3, 4]
    enc_v1 = ts.ckks_vector(context, v1)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    remove_ctx = ctx_ptr.get(delete_obj=False)
    enc_v1 = enc_v1_ptr.get(delete_obj=False)

    enc_v1.link_context(remove_ctx)


@pytest.mark.vendor(lib="tenseal")
def test_context_link_ptr(context: Any, root_client: sy.VirtualMachineClient) -> None:
    v1 = [0, 1, 2, 3, 4]
    enc_v1 = ts.ckks_vector(context, v1)

    ctx_ptr = context.send(root_client, pointable=True)
    enc_v1_ptr = enc_v1.send(root_client, pointable=True)

    assert ctx_ptr.is_private().get()
    assert not ctx_ptr.has_galois_keys().get()
    assert ctx_ptr.has_secret_key().get()
    assert ctx_ptr.has_public_key().get()
    assert ctx_ptr.has_relin_keys().get()

    enc_v1_ptr.link_context(ctx_ptr)

    result = enc_v1_ptr.get()
    result.link_context(context)
    result = result.decrypt()

    assert pytest.approx(result, abs=0.001) == [0, 1, 2, 3, 4]


@pytest.mark.vendor(lib="tenseal")
def test_context_generate_relin_keys(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    context.generate_relin_keys()
    ctx_ptr = context.send(root_client, pointable=True)

    assert ctx_ptr.has_relin_keys().get()


@pytest.mark.vendor(lib="tenseal")
def test_context_generate_galois_keys(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    context.generate_galois_keys()
    ctx_ptr = context.send(root_client, pointable=True)

    assert ctx_ptr.has_galois_keys().get()


@pytest.mark.vendor(lib="tenseal")
def test_context_make_public(
    context: Any, root_client: sy.VirtualMachineClient
) -> None:
    context.make_context_public(generate_galois_keys=False, generate_relin_keys=False)

    ctx_ptr = context.send(root_client, pointable=True)

    assert not ctx_ptr.is_private().get()
    assert not ctx_ptr.has_galois_keys().get()
    assert not ctx_ptr.has_secret_key().get()
    assert ctx_ptr.has_public_key().get()
    assert ctx_ptr.has_relin_keys().get()
    assert ctx_ptr.is_public().get()


@pytest.mark.vendor(lib="tenseal")
def test_context_options(context: Any, root_client: sy.VirtualMachineClient) -> None:
    ctx_ptr = context.send(root_client, pointable=True)

    assert ctx_ptr.auto_mod_switch.get()
    assert ctx_ptr.auto_relin.get()
    assert ctx_ptr.auto_rescale.get()
    assert ctx_ptr.global_scale.get() == 2 ** 40
