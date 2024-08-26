# stdlib

# third party
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from .code_history import CodeHistory

NamePartitionKey = PartitionKey(key="service_func_name", type_=str)
VerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)


@serializable(canonical_name="CodeHistoryStashSQL", version=1)
class CodeHistoryStash(ObjectStash[CodeHistory]):
    settings: PartitionSettings = PartitionSettings(
        name=CodeHistory.__canonical_name__, object_type=CodeHistory
    )

    def get_by_service_func_name_and_verify_key(
        self,
        credentials: SyftVerifyKey,
        service_func_name: str,
        user_verify_key: SyftVerifyKey,
    ) -> Result[list[CodeHistory], str]:
        return self.get_one_by_fields(
            credentials=credentials,
            fields={
                "user_verify_key": str(user_verify_key),
                "service_func_name": service_func_name,
            },
        )

    def get_by_service_func_name(
        self, credentials: SyftVerifyKey, service_func_name: str
    ) -> Result[list[CodeHistory], str]:
        return self.get_all_by_field(
            credentials=credentials,
            field_name="service_func_name",
            field_value=service_func_name,
        )

    def get_by_verify_key(
        self, credentials: SyftVerifyKey, user_verify_key: SyftVerifyKey
    ) -> Result[CodeHistory | None, str]:
        return self.get_all_by_field(
            credentials=credentials,
            field_name="user_verify_key",
            field_value=str(user_verify_key),
        )
