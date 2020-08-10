# external class imports
from typing import Dict
from typing import Optional
from typing_extensions import final

# syft imports
from .device_type.device_type import DeviceType
from .device_type.unknown import unknown_device
from ..vm.client import VirtualMachineClient
from ...io.location import SpecificLocation
from ...common.message import SyftMessage
from ....decorators import syft_decorator
from syft.core.common.uid import UID
from syft.core.io.address import All
from ...io.location import Location
from ..vm.vm import VirtualMachine
from .client import DeviceClient
from ..common.node import Node


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
        device: SpecificLocation = SpecificLocation(),
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
    def id(self) -> UID:
        return self.device.id if self.device is not None else None

    @syft_decorator(typechecking=True)
    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return msg.address.device.id in (self.id, All(),) and msg.address.vm is None
