# relative
from ..serde.serializable import serializable
from .node import Node


@serializable(without=["queue_manager"])
class Datasite(Node):
    pass
