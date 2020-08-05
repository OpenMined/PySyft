from typing import Dict
from typing_extensions import final

from syft.core.common.message import SyftMessage
from syft.core.common.uid import UID

from ....decorators import syft_decorator
from ..common.node import Node
from ..vm.client import VirtualMachineClient
from ..vm.vm import VirtualMachine
from .client import DeviceClient
from .device_type.device_type import DeviceType
from .device_type.unknown import unknown_device


@final
class Device(Node):

    client_type = DeviceClient

    child_type = VirtualMachine
    child_type_client_type = VirtualMachineClient

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: str,
        device_type: DeviceType = unknown_device,
        vms: Dict[UID, VirtualMachine] = {},
    ):
        super().__init__(name=name)

        self.device_type = device_type

        self._register_services()

    def add_me_to_my_address(self):
        self.address.pri_address.device = self.id

    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return (
            msg.address.pri_address.device in (self.id, All())
            and msg.address.pri_address.vm in (None, Unspecified())
        )
