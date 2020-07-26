from ..abstract.node import Node
from ...store.storeable.dataset import Dataset
from ...store.storeable.model import Model
from ...message.syft_message import SyftMessage
from typing import List
from ..domain.domain import Domain
from ..domain.client import DomainClient
from .client import NetworkClient



class Network(Node):

    client_type = NetworkClient

    child_type = Domain
    child_type_client_type = DomainClient

    def __init__(self, name: str):
        super().__init__(name=name)

        self._register_services()

    def add_me_to_my_address(self):
        self.address.pub_address.network = self.id

    def message_is_for_me(self, msg: SyftMessage) -> bool:
        return (
            msg.address.pub_address.network == self.id
            and msg.address.pub_address.domain is None
        )