# syft absolute
import syft as sy
from syft.core.node.common.service.child_node_lifecycle_service import (
    RegisterChildNodeMessage,
)


def test_child_node_lifecycle_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    # bob_phone_client.register(client=bob_vm_client)
    # generates this message
    msg = RegisterChildNodeMessage(
        lookup_id=bob_vm_client.id,  # TODO: not sure if this is needed anymore
        child_node_client_address=bob_vm_client.address,
        address=bob_phone_client.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == msg2.address
    assert msg.child_node_client_address == msg2.child_node_client_address
    assert msg == msg2
