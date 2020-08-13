import syft as sy

from syft.core.io.address import Address
from syft.core.common.uid import UID
from syft.core.node.common.service.child_node_lifecycle_service import (
    RegisterChildNodeMessage,
)


def test_child_node_lifecycle_message_serde():
    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    # bob_phone_client.register(client=bob_vm_client)
    # generates this message
    msg = RegisterChildNodeMessage(
        child_node_client=bob_vm_client, address=bob_phone_client
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == msg2.address
    assert msg.child_node_client == msg2.child_node_client
    assert msg == msg2
