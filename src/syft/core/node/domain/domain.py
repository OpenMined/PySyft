# external class imports
from typing import Optional, Union
from nacl.signing import VerifyKey

# syft imports
from ....decorators.syft_decorator_impl import syft_decorator
from ...io.location import SpecificLocation
from ...common.message import SyftMessage, SignedMessage
from ..device import Device, DeviceClient
from ...io.location import Location
from .client import DomainClient
from ..common.node import Node
from ...common.uid import UID
from ...io.address import All


class Domain(Node):

    domain: SpecificLocation
    root_key: Optional[VerifyKey]

    child_type = Device
    client_type = DomainClient
    child_type_client_type = DeviceClient

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: str,
        network: Optional[Location] = None,
        domain: SpecificLocation = SpecificLocation(),
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
        root_key: Optional[VerifyKey] = None,
    ):
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )

        self.root_key = root_key

        # available_device_types = set()
        # TODO: add available compute types

        # default_device = None
        # TODO: add default compute type

        self._register_services()

    @property
    def id(self) -> UID:
        return self.domain.id

    def message_is_for_me(self, msg: Union[SyftMessage, SignedMessage]) -> bool:
        return msg.address.domain.id in (self.id, All(),) and msg.address.device is None
