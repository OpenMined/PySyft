# third party
import pytest
from typing import Any
from typing import Sequence

# syft absolute
import syft as sy


def _almost_equal(vec1: Sequence, vec2: Sequence, precision_pow_ten: int = 1) -> bool:
    if len(vec1) != len(vec2):
        return False

    upper_bound = pow(10, -precision_pow_ten)
    for v1, v2 in zip(vec1, vec2):
        if abs(v1 - v2) > upper_bound:
            return False
    return True


@pytest.fixture(scope="function")
def context() -> Any:
    import tenseal as ts

    sy.load_lib("tenseal")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 16384, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
    )
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()
    return context


@pytest.fixture(scope="function")
def duet() -> Any:
    return sy.VirtualMachine().get_root_client()


@pytest.mark.vendor(lib="tenseal")
def test_context_send(context: Any) -> None:
    """Test sending a TenSEAL context"""
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    assert len(alice.store) == 0

    _ = context.send(alice_client)

    assert len(alice.store) == 1


@pytest.mark.vendor(lib="tenseal")
def test_context_link(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    v1 = [0, 1, 2, 3, 4]
    enc_v1 = ts.ckks_vector(context, v1)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    remove_ctx = ctx_ptr.get(delete_obj=False)
    enc_v1 = enc_v1_ptr.get(delete_obj=False)

    enc_v1.link_context(remove_ctx)


@pytest.mark.vendor(lib="tenseal")
def test_context_link_ptr(context: Any, duet: sy.VirtualMachine) -> None:
    import tenseal as ts

    v1 = [0, 1, 2, 3, 4]
    enc_v1 = ts.ckks_vector(context, v1)

    ctx_ptr = context.send(duet, searchable=True)
    enc_v1_ptr = enc_v1.send(duet, searchable=True)

    enc_v1_ptr.link_context(ctx_ptr)

    result = enc_v1_ptr.decrypt().get()
    assert _almost_equal(result, [0, 1, 2, 3, 4])
