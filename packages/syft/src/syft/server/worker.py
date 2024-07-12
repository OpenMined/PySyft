# relative
from ..serde.serializable import serializable
from .server import Server


@serializable()
class Worker(Server):
    pass
