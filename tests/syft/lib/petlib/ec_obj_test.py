# Testing code

# third party
import petlib
from petlib.ec import EcGroup
import pytest

# syft absolute
import syft as sy


@pytest.fixture(scope="function")
def ec_group() -> petlib.ec.EcGroup:
    eg = petlib.ec.EcGroup()
    return eg


def test_protobuf_Ec_serializer_deserializer(ec_group: EcGroup) -> None:
    # third party
    import petlib

    sy.load("petlib")
    vm = sy.VirtualMachine()
    client = vm.get_root_client()

    remote_ec_group = ec_group.send(client)
    recieved_ec_group = remote_ec_group.get()

    assert recieved_ec_group == ec_group

    ec_pt = petlib.ec.EcPt(ec_group)
    remote_ec_pt = ec_pt.send(client)
    recieved_ec_pt = remote_ec_pt.get()

    assert recieved_ec_pt == ec_pt
