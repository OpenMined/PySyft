from ..common.node import Node
from typing import final
from .client import DeviceClient
from ....decorators import syft_decorator
from syft.core.common.uid import UID
from .device_type.device_type import DeviceType
from .device_type.unknown import unknown_device
from ..vm.vm import VirtualMachine
from ..vm.client import VirtualMachineClient
from typing import Dict
from syft.core.message import SyftMessage


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

    def get_vm(self, id_or_name: (str, UID)):
        try:
            return self.vms[id_or_name]
        except KeyError as e:
            try:
                id = self.vm_name2id[id_or_name]
                return self.vms[id]
            except KeyError as e:
                raise KeyError("You must ask for a vm using either a name or ID.")

    @syft_decorator(typechecking=True)
    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return (
            msg.address.pri_address.device == self.id
            and msg.address.pri_address.vm is None
        )
