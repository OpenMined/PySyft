# stdlib

# third party
from result import Result
from syft.service.code_history.code_history_sql import CodeHistoryDB
from syft.service.job.job_sql_stash import ObjectStash

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from .code_history import CodeHistory

NamePartitionKey = PartitionKey(key="service_func_name", type_=str)
VerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)


@serializable(canonical_name="CodeHistoryStashSQL", version=1)
class CodeHistoryStashSQL(ObjectStash[CodeHistory, CodeHistoryDB]):
    object_type = CodeHistory
    settings: PartitionSettings = PartitionSettings(
        name=CodeHistory.__canonical_name__, object_type=CodeHistory
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_service_func_name_and_verify_key(
        self,
        credentials: SyftVerifyKey,
        service_func_name: str,
        user_verify_key: SyftVerifyKey,
    ) -> Result[list[CodeHistory], str]:
        return self.get_one_by_property(
            credentials=user_verify_key,
            property_name="service_func_name",
            property_value=service_func_name,
        )

    def get_by_service_func_name(
        self, credentials: SyftVerifyKey, service_func_name: str
    ) -> Result[list[CodeHistory], str]:
        return self.get_many_by_property(
            credentials=credentials,
            property_name="service_func_name",
            property_value=service_func_name,
        )

    def get_by_verify_key(
        self, credentials: SyftVerifyKey, user_verify_key: SyftVerifyKey
    ) -> Result[CodeHistory | None, str]:
        if isinstance(user_verify_key, str):
            user_verify_key = SyftVerifyKey.from_string(user_verify_key)
        return self.get_all(user_verify_key)
