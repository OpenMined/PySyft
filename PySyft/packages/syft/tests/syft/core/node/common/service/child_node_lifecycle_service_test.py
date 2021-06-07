# third party
import pytest

# syft absolute
import syft as sy
from syft import serialize
from syft.core.node.common.service.child_node_lifecycle_service import (
    RegisterChildNodeMessage,
)


@pytest.mark.slow
def test_child_node_lifecycle_message_serde(
    node: sy.VirtualMachine, client: sy.VirtualMachineClient
) -> None:
    second_client = node.get_client()

    # bob_phone_client.register(client=bob_vm_client)
    # generates this message
    msg = RegisterChildNodeMessage(
        lookup_id=client.id,  # TODO: not sure if this is needed anymore
        child_node_client_address=client.address,
        address=second_client.address,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == msg2.address
    assert msg.child_node_client_address == msg2.child_node_client_address
    assert msg == msg2
