from syft.core.common.message import SyftMessage

from ..common.node import Node
from ..domain.client import DomainClient
from ..domain.domain import Domain
from .client import NetworkClient


class Network(Node):

    client_type = NetworkClient

    child_type = Domain
    child_type_client_type = DomainClient

    def __init__(self, name: str):
        super().__init__(name=name)

        self._register_services()

    def add_me_to_my_address(self) -> None:
        self.address.pub_address.network = self.id

    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return msg.address.pub_address.network in (
            self.id,
            All(),
        ) and msg.address.pub_address.domain in (None, Unspecified())
