# stdlib

# third party
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...util.telemetry import instrument
from .user_code import UserCode


@instrument
@serializable(canonical_name="UserCodeSQLStash", version=1)
class UserCodeStash(ObjectStash[UserCode]):
    object_type = UserCode
    settings: PartitionSettings = PartitionSettings(
        name=UserCode.__canonical_name__, object_type=UserCode
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_code_hash(
        self, credentials: SyftVerifyKey, code_hash: str
    ) -> Result[UserCode | None, str]:
        return self.get_one_by_field(
            credentials=credentials,
            field_name="code_hash",
            field_value=code_hash,
        )

    def get_by_service_func_name(
        self, credentials: SyftVerifyKey, service_func_name: str
    ) -> Result[list[UserCode], str]:
        return self.get_all_by_field(
            credentials=credentials,
            field_name="service_func_name",
            field_value=service_func_name,
        )
