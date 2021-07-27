# third party
import pytest

# syft absolute
import syft as sy
import syft_petlib  # noqa :401

petlib = pytest.importorskip("petlib")

@pytest.mark.vendor(lib="petlib")
def test_ecpt_serde(root_client:sy.VirtualMachineClient) -> None:
    ec_group = petlib.ec.EcGroup()

    remote_ec_group = ec_group.send(root_client)
    received_ec_group = remote_ec_group.get()

    assert received_ec_group == ec_group

    ec_pt = petlib.ec.EcPt(ec_group)

    remote_ec_pt = ec_pt.send(root_client)
    received_ec_group = remote_ec_pt.get()

    assert received_ec_group == ec_pt

    bn = petlib.bn.Bn(1)

    bn_ptr = bn.send(root_client)
    bn2 = bn_ptr.get()

    assert bn == bn2
