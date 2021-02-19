# stdlib
from typing import Optional
from typing import Union

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft relative
from ....logger import error
from ...common.message import SignedMessage
from ...common.message import SyftMessage
from ...common.uid import UID
from ...io.location import Location
from ...io.location import SpecificLocation
from ..common.node import Node
from ..domain.client import DomainClient
from ..domain.domain import Domain
from .client import NetworkClient


class Network(Node):

    network: SpecificLocation

    child_type = Domain
    client_type = NetworkClient
    child_type_client_type = DomainClient

    def __init__(
        self,
        name: Optional[str],
        network: SpecificLocation = SpecificLocation(),
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: Optional[Location] = None,
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
        self.network = SpecificLocation(name=self.name)

        self._register_services()
        self.post_init()

    @property
    def icon(self) -> str:
        return "🔗"

    @property
    def id(self) -> UID:
        return self.network.id

    def message_is_for_me(self, msg: Union[SyftMessage, SignedMessage]) -> bool:
        # this needs to be defensive by checking network_id NOT network.id or it breaks
        try:
            return msg.address.network_id == self.id and msg.address.domain is None
        except Exception as e:
            error(f"Error checking if {msg.pprint} is for me on {self.pprint}. {e}")
            return False
