# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from .user_code import CodeHashPartitionKey
from .user_code import ServiceFuncNamePartitionKey
from .user_code import SubmitTimePartitionKey
from .user_code import UserCode
from .user_code import UserVerifyKeyPartitionKey


@serializable(canonical_name="UserCodeStash", version=1)
class UserCodeStash(NewBaseUIDStoreStash):
    object_type = UserCode
    settings: PartitionSettings = PartitionSettings(
        name=UserCode.__canonical_name__, object_type=UserCode
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    @as_result(StashException, NotFoundException)
    def get_all_by_user_verify_key(
        self, credentials: SyftVerifyKey, user_verify_key: SyftVerifyKey
    ) -> list[UserCode]:
        qks = QueryKeys(qks=[UserVerifyKeyPartitionKey.with_obj(user_verify_key)])
        return self.query_one(credentials=credentials, qks=qks).unwrap()

    @as_result(StashException, NotFoundException)
    def get_by_code_hash(self, credentials: SyftVerifyKey, code_hash: str) -> UserCode:
        qks = QueryKeys(qks=[CodeHashPartitionKey.with_obj(code_hash)])
        return self.query_one(credentials=credentials, qks=qks).unwrap()

    @as_result(StashException)
    def get_by_service_func_name(
        self, credentials: SyftVerifyKey, service_func_name: str
    ) -> list[UserCode]:
        qks = QueryKeys(qks=[ServiceFuncNamePartitionKey.with_obj(service_func_name)])
        return self.query_all(
            credentials=credentials, qks=qks, order_by=SubmitTimePartitionKey
        ).unwrap()
