from typing import Dict
from typing_extensions import final

from syft.core.common.message import SyftMessage
from syft.core.common.uid import UID
from syft.core.io.address import All

from ....decorators import syft_decorator
from ..common.node import Node
from ..vm.client import VirtualMachineClient
from ..vm.vm import VirtualMachine
from .client import DeviceClient
from .device_type.device_type import DeviceType
from .device_type.unknown import unknown_device

from typing import Optional
from ...io.location import Location
from ...io.location import SpecificLocation


@final
class Device(Node):

    client_type = DeviceClient

    child_type = VirtualMachine
    child_type_client_type = VirtualMachineClient

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: str,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[SpecificLocation] = SpecificLocation(),
        vm: Optional[Location] = None,
        device_type: DeviceType = unknown_device,
        vms: Dict[UID, VirtualMachine] = {},
    ):
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )

        self.device_type = device_type

        self._register_services()

    @property
    def id(self):
        return self.device.id

    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return msg.address.device.id in (self.id, All(),) and msg.address.vm is None
