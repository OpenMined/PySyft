from .device_message import DeviceMessage
from ....io.route import Route
from .....common.id import UID
from ...vm.vm import VirtualMachineClient


class VirtualMachineLifecycleMessage(DeviceMessage):
    """An abstract class for all VM Lifecycle classes to extend"""


class CreateVirtualMachineMessage(VirtualMachineLifecycleMessage):
    def __init__(self, vm_name: str, route: Route, msg_id: UID = None):
        super().__init__(route=route, msg_id=msg_id)
        self.vm_name = vm_name


class CreateVirtualMachineReplyMessage(VirtualMachineLifecycleMessage):
    def __init__(self, client: VirtualMachineClient, route: Route, msg_id: UID = None):
        super().__init__(route=route, msg_id=msg_id)
        self.client = client
