# relative
from ..serde.serializable import serializable
from .server import Server


@serializable(without=["queue_manager"], canonical_name="Datasite", version=1)
class Datasite(Server):
    pass
