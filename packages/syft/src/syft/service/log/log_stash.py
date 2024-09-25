# relative
from ...serde.serializable import serializable
from ...store.db.stash import ObjectStash
from .log import SyftLog


@serializable(canonical_name="LogStash", version=1)
class LogStash(ObjectStash[SyftLog]):
    pass
