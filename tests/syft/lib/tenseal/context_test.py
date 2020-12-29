# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="tenseal")
def test_context_send() -> None:
    # third party
    import tenseal as ts

    """Test sending a TenSEAL context"""

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    assert len(alice.store) == 0

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, 0, [40, 20, 40])
    ctx.global_scale = 2 ** 40

    sy.load_lib("tenseal")

    _ = ctx.tag("context").send(alice_client)

    assert len(alice.store) == 1


@pytest.mark.vendor(lib="tenseal")
@pytest.mark.parametrize(
    "property_name", ["global_scale", "auto_mod_switch", "auto_relin", "auto_rescale"]
)
def test_context_property(property_name: str) -> None:
    import tenseal as ts

    sy.load_lib("tenseal")
    remote = sy.VirtualMachine().get_root_client()

    context = ts.Context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40

    local_get_property = getattr(context, property_name)

    context_ptr = context.send(remote)

    # access the property remotely, get it's pointer
    property_ptr = getattr(context_ptr, property_name)

    # get back the property value from the remote side
    # test if it's the same as what we got locally
    assert property_ptr.get() == local_get_property
