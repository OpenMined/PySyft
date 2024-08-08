# relative
from syft.service.job.job_sql_stash import ObjectStash
from syft.service.log.log_sql import SyftLogDB
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...util.telemetry import instrument
from .log import SyftLog


@instrument
@serializable(canonical_name="LogStashSQL", version=1)
class LogStashSQL(ObjectStash[SyftLog, SyftLogDB]):
    object_type = SyftLog
    settings: PartitionSettings = PartitionSettings(
        name=SyftLog.__canonical_name__, object_type=SyftLog
    )
