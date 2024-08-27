# relative
from ...serde.serializable import serializable
from ...store.db.stash import ObjectStash
from ...util.telemetry import instrument
from .log import SyftLog


@instrument
@serializable(canonical_name="LogStash", version=1)
class LogStash(ObjectStash[SyftLog]):
    pass
