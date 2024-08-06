from syft.service.code.user_code import UserCode
from syft.service.code.user_code_sql import UserCodeDB
from syft.service.job.job_sql_stash import ObjectStash

# relative
from ...serde.serializable import serializable
from ...store.document_store import PartitionSettings


@serializable(canonical_name="UserCodeStashSQL", version=1)
class UserCodeStashSQL(ObjectStash[UserCode, UserCodeDB]):
    object_type = UserCode
    settings: PartitionSettings = PartitionSettings(
        name=UserCode.__canonical_name__, object_type=UserCode
    )

    def __init__(self, server_uid) -> None:
        super().__init__(server_uid, UserCode, UserCodeDB)
