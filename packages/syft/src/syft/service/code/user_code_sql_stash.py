# relative
from ...serde.serializable import serializable
from ...store.document_store import PartitionSettings
from ..job.job_sql_stash import ObjectStash
from .user_code import UserCode
from .user_code_sql import UserCodeDB


@serializable(canonical_name="UserCodeStashSQL", version=1)
class UserCodeStashSQL(ObjectStash[UserCode, UserCodeDB]):
    object_type = UserCode
    settings: PartitionSettings = PartitionSettings(
        name=UserCode.__canonical_name__, object_type=UserCode
    )

    def __init__(self, store) -> None:
        super().__init__(store=store)
