# stdlib
from typing import Any
from typing import Optional

# third party
from nacl.signing import SigningKey
from typing_extensions import final

# relative
from ..common.uid import UID
from ..io.location import Location
from ..io.location import SpecificLocation
from .common.node import Node
from .common.node_manager.dict_store import DictStore
from .device_client import DeviceClient
from .vm import VirtualMachine
from .vm_client import VirtualMachineClient


@final
class Device(Node):

    device: SpecificLocation

    client_type = DeviceClient
    child_type = VirtualMachine
    child_type_client_type = VirtualMachineClient

    def __init__(
        self,
        name: Optional[str] = None,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: SpecificLocation = SpecificLocation(),
        vm: Optional[Location] = None,
        device_type: Any = None,
        signing_key: Optional[SigningKey] = None,
        store_type: type = DictStore,
    ):
        super().__init__(
            name=name,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            store_type=store_type,
        )

        # specific location with name
        self.device = SpecificLocation(name=self.name)

        self.device_type = device_type

        self._register_services()

        self.post_init()

    def post_init(self) -> None:
        super().post_init()

    @property
    def icon(self) -> str:
        return "📱"

    @property
    def id(self) -> UID:
        return self.node_uid
