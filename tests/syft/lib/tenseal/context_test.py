# third party
import pytest

# syft absolute
import syft as sy

# third party
import tenseal as ts


@pytest.mark.vendor(lib="tenseal")
def test_context_send() -> None:
    """Test sending a TenSEAL context"""

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    assert len(alice.store) == 0

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, 0, [40, 20, 40])
    ctx.global_scale = 2 ** 40

    sy.load_lib("tenseal")

    _ = ctx.tag("context").send(alice_client)

    assert len(alice.store) == 1
