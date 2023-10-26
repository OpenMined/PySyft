# relative
from ..abstract_node import NodeType
from ..serde.serializable import serializable
from .node import Node


@serializable()
class Gateway(Node):
    def post_init(self) -> None:
        self.node_type = NodeType.GATEWAY
        super().post_init()
