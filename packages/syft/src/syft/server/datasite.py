# relative
from ..serde.serializable import serializable
from .server import Server


@serializable(without=["queue_manager"])
class Datasite(Server):
    pass
