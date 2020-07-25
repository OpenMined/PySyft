from ..abstract.node import Node
from ...store.storeable.dataset import Dataset
from ...store.storeable.model import Model
from ...message.syft_message import SyftMessage
from typing import List
from ..domain.domain import Domain
from ..domain.client import DomainClient

class Network(Node):

    child_type = Domain
    child_type_client_type = DomainClient

    def __init__(self, name: str, domains: List[Domain]):
        super().__init__(name=name)

    def message_is_for_me(self, msg:SyftMessage) -> bool:
        return msg.address.pub_address.network == self.id and msg.address.pub_address.domain is None

    def _register_services(self) -> None:
        services = list()

        for s in services:
            self.msg_router[s.message_handler_type()] = s()
