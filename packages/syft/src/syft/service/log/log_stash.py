# relative
from ...serde.serializable import serializable
from ...store.db.stash import ObjectStash
from ...store.document_store import PartitionSettings
from ...util.telemetry import instrument
from .log import SyftLog


@instrument
@serializable(canonical_name="LogStash", version=1)
class LogStash(ObjectStash[SyftLog]):
    settings: PartitionSettings = PartitionSettings(
        name=SyftLog.__canonical_name__, object_type=SyftLog
    )
