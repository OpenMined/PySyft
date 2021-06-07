# third party
import pytest

# syft absolute
import syft as sy

petlib = pytest.importorskip("petlib")
sy.load("petlib")


@pytest.mark.vendor(lib="petlib")
def test_ecpt_serde() -> None:
    vm = sy.VirtualMachine()
    client = vm.get_root_client()
    ec_group = petlib.ec.EcGroup()

    remote_ec_group = ec_group.send(client)
    received_ec_group = remote_ec_group.get()

    assert received_ec_group == ec_group

    ec_pt = petlib.ec.EcPt(ec_group)

    remote_ec_pt = ec_pt.send(client)
    received_ec_group = remote_ec_pt.get()

    assert received_ec_group == ec_pt

    bn = petlib.bn.Bn(1)

    bn_ptr = bn.send(client)
    bn2 = bn_ptr.get()

    assert bn == bn2
