# relative
from ..serde.serializable import serializable
from .node import Node


@serializable()
class Worker(Node):
    pass
