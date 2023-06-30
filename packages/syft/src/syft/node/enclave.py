# relative
from ..abstract_node import NodeType
from ..serde.serializable import serializable
from .node import Node


@serializable()
class Enclave(Node):
    def post_init(self) -> None:
        self.node_type = NodeType.ENCLAVE
        super().post_init()
