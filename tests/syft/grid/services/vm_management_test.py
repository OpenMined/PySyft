# syft absolute
import syft as sy
from syft.core.node.device.service.vm_management_message import CreateVMMessage
from syft.core.node.device.service.vm_management_message import CreateVMResponseMessage


def test_create_vm_message_serde() -> None:
    bob_domain = sy.Domain(name="Bob")
    target_device = sy.Device(name="Common Device")

    settings = {"cpu": "xeon", "gpu": "Tesla", "memory": "32gb"}
    msg = CreateVMMessage(address=target_device, settings=settings, reply_to=bob_domain)

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target_device.address
    assert msg.settings == msg2.settings
    assert msg == msg2


def test_create_vm_response_message_serde() -> None:
    bob_domain = sy.Domain(name="Bob")

    new_vm = sy.VirtualMachine(name="Custom Virtual Machine")

    msg = CreateVMResponseMessage(
        address=bob_domain,
        success=True,
        msg="Virtual Machine initialized successfully!",
        vm_address=new_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == bob_domain.address
    assert msg.vm_address == new_vm.address
    assert msg == msg2
    assert msg.msg == msg.msg
    assert msg.success == msg.success
