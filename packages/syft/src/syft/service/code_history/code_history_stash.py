# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import StashException
from ...types.result import as_result
from .code_history import CodeHistory


@serializable(canonical_name="CodeHistoryStashSQL", version=1)
class CodeHistoryStash(ObjectStash[CodeHistory]):
    @as_result(StashException)
    def get_by_service_func_name_and_verify_key(
        self,
        credentials: SyftVerifyKey,
        service_func_name: str,
        user_verify_key: SyftVerifyKey,
    ) -> CodeHistory:
        return self.get_one_by_fields(
            credentials=credentials,
            fields={
                "user_verify_key": str(user_verify_key),
                "service_func_name": service_func_name,
            },
        ).unwrap()

    @as_result(StashException)
    def get_by_service_func_name(
        self, credentials: SyftVerifyKey, service_func_name: str
    ) -> list[CodeHistory]:
        return self.get_all_by_field(
            credentials=credentials,
            field_name="service_func_name",
            field_value=service_func_name,
        )

    @as_result(StashException)
    def get_by_verify_key(
        self, credentials: SyftVerifyKey, user_verify_key: SyftVerifyKey
    ) -> list[CodeHistory]:
        return self.get_all_by_field(
            credentials=credentials,
            field_name="user_verify_key",
            field_value=str(user_verify_key),
        ).unwrap()
