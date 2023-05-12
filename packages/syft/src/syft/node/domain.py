# relative
from ..serde.serializable import serializable
from .node import Node


@serializable(without=["queue_router", "publisher"])
class Domain(Node):
    pass
