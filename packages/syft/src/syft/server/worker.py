# relative
from ..serde.serializable import serializable
from .server import Server


@serializable(canonical_name="Worker", version=1)
class Worker(Server):
    pass
