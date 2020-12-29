# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="tenseal")
@pytest.mark.parametrize("member_name", ["BFV", "CKKS", "NONE"])
def test_scheme_type(member_name: str) -> None:
    # third party
    import tenseal as ts

    sy.load_lib("tenseal")

    remote = sy.VirtualMachine().get_root_client()

    # create a SCHEME_TYPE object
    scheme_type = getattr(ts.SCHEME_TYPE, member_name)

    # send the SCHEME_TYPE object
    tags = ["scheme type"]
    description = "the scheme type we want"
    scheme_type_ptr = scheme_type.send(remote, tags=tags, description=description)

    # test if we send it well
    assert len(remote.store) == 1
    assert scheme_type_ptr.tags == tags
    assert scheme_type_ptr.description == description

    # test if we can get it well
    assert scheme_type_ptr.get() == scheme_type
