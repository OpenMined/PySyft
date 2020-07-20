from ..abstract.client import Client
from typing import final
from .message.lifecycles_messages import CreateVirtualMachineMessage
from ...io.route import route


@final
class DeviceClient(Client):
    def __init__(self, device_id, name, connection):
        super().__init__(worker_id=device_id, name=name, connection=connection)

    def create_vm(self, name: str):

        # Step 1: create a message which will request for a VM to be created.
        # we can set route=None because we know the message is just going directly
        # to the device.
        msg = CreateVirtualMachineMessage(vm_name=name, route=None)

        # Step 2: Send the message to the device this client points to.
        # Receive a reply_msg and save it in a variable.
        reply_msg = self.send_msg(msg=msg)

        # Step 3: Unpack the vm client from the reply message
        vm_client = reply_msg.client

        # Step 4: Return the client object directly. We assume that the
        # client object knows how to send message to the VM no matter
        # where it exists. Aka, we don't need to save the client in any
        # particular kind of context to know how to use it. We should be
        # able to "just use it" anywhere and it should "just work"... finding
        # the appropriate VM.
        return vm_client

    def __repr__(self):
        out = f"<DeviceClient id:{self.worker_id}>"
        return out
