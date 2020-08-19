# external class imports
from typing import Dict, Optional, Union
from typing_extensions import final

from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft imports
from .device_type.device_type import DeviceType
from .device_type.unknown import unknown_device
from ..vm.client import VirtualMachineClient
from ...io.location import SpecificLocation
from ...common.message import SyftMessage
from ...common.message import SignedMessage
from ....decorators import syft_decorator
from syft.core.common.uid import UID
from ...io.location import Location
from ..vm.vm import VirtualMachine
from .client import DeviceClient
from ..common.node import Node


@final
class Device(Node):

    device: SpecificLocation

    client_type = DeviceClient
    child_type = VirtualMachine
    child_type_client_type = VirtualMachineClient

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: Optional[str] = None,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: SpecificLocation = SpecificLocation(),
        vm: Optional[Location] = None,
        device_type: DeviceType = unknown_device,
        vms: Dict[UID, VirtualMachine] = {},
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
    ):
        super().__init__(
            name=name,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
        )

        # specific location with name
        self.device = SpecificLocation(name=self.name)

        self.device_type = device_type

        self._register_services()

        self.post_init()

    @property
    def icon(self) -> str:
        return "📱"

    @property
    def id(self) -> UID:
        return self.device.id

    def message_is_for_me(self, msg: Union[SyftMessage, SignedMessage]) -> bool:
        # this needs to be defensive by checking device_id NOT device.id or it breaks
        try:
            return msg.address.device_id == self.id and msg.address.vm is None
        except Exception as e:
            error = f"Error checking if {msg.pprint} is for me on {self.pprint}. {e}"
            print(error)
            return False
