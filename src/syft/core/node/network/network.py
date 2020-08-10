# external class imports
from typing import Optional

# syft imports (sorted by length)
from ....decorators.syft_decorator_impl import syft_decorator
from ...io.location import SpecificLocation
from ...common.message import SyftMessage
from ..domain.client import DomainClient
from ..domain.domain import Domain
from ...io.location import Location
from .client import NetworkClient
from ..common.node import Node
from ...io.address import All
from ...common.uid import UID


class Network(Node):

    network: SpecificLocation

    child_type = Domain
    client_type = NetworkClient
    child_type_client_type = DomainClient

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: str,
        network: SpecificLocation = SpecificLocation(),
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
    ):
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )

        self._register_services()

    @property
    def id(self) -> UID:
        return self.network.id

    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return (
            msg.address.network.id in (self.id, All(),) and msg.address.domain is None
        )
