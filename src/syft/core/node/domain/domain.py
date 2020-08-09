from syft.core.common.message import SyftMessage

from ..common.node import Node
from ..device import Device, DeviceClient
from .client import DomainClient
from syft.core.io.address import All, Unspecified

from typing import Optional
from ...io.location import Location
from ...io.location import SpecificLocation
from ....decorators.syft_decorator_impl import syft_decorator

class Domain(Node):

    client_type = DomainClient

    child_type = Device
    child_type_client_type = DeviceClient

    @syft_decorator(typechecking=True)
    def __init__(self,
                 name:str,
                 network: Optional[Location] = None,
                 domain: Optional[SpecificLocation] = SpecificLocation(),
                 device: Optional[Location] = None,
                 vm: Optional[Location] = None):
        super().__init__(name=name,
                         network=network,
                         domain=domain,
                         device=device,
                         vm=vm)

        # available_device_types = set()
        # TODO: add available compute types

        # default_device = None
        # TODO: add default compute type

        self._register_services()

    @property
    def id(self):
        return self.domain.id

    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return msg.address.domain.id in (
            self.id,
            All(),
        ) and msg.address.device is None
