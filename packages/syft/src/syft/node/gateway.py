# relative
from ..abstract_node import NodeType
from ..serde.serializable import serializable
from ..service.context import AuthedServiceContext
from ..service.network.network_service import NetworkService
from .node import Node


@serializable()
class Gateway(Node):
    def post_init(self) -> None:
        self.node_type = NodeType.GATEWAY
        super().post_init()
        self.connect_to_vpn_self()

    def connect_to_vpn_self(self) -> None:
        network_service = self.get_service(NetworkService)
        context = AuthedServiceContext(
            node=self, credentials=self.signing_key.verify_key
        )
        result = network_service.connect_self(context=context)
        print("Message: ", result.message)
