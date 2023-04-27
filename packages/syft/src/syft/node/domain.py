# relative
from ..serde.serializable import serializable
from .node import Node


@serializable()
class Domain(Node):
    pass
