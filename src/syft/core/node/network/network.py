from syft.core.common.message import SyftMessage
from syft.core.io.address import All, Unspecified

from ..common.node import Node
from ..domain.client import DomainClient
from ..domain.domain import Domain
from .client import NetworkClient

from typing import Optional
from ...io.location import Location
from ...io.location import SpecificLocation
from ....decorators.syft_decorator_impl import syft_decorator

class Network(Node):

    client_type = NetworkClient

    child_type = Domain
    child_type_client_type = DomainClient

    @syft_decorator(typechecking=True)
    def __init__(self,
                 name: str,
                 network: Optional[SpecificLocation] = SpecificLocation,
                 domain: Optional[Location] = None,
                 device: Optional[Location] = None,
                 vm: Optional[Location] = None):
        super().__init__(name=name,
                         network=network,
                         domain=domain,
                         device=device,
                         vm=vm)

        self._register_services()

    @property
    def id(self):
        return self.network.id

    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return msg.address.pub_address.network.id in (
            self.id,
            All(),
        ) and msg.address.pub_address.domain in (None, Unspecified())
