# relative
from ..serde.serializable import serializable
from .node import Node


@serializable()
class Gateway(Node):
    pass
