# stdlib
from typing import List
from typing import Optional

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...util.telemetry import instrument
from .user_code import CodeHashPartitionKey
from .user_code import ServiceFuncNamePartitionKey
from .user_code import SubmitTimePartitionKey
from .user_code import UserCode
from .user_code import UserVerifyKeyPartitionKey


@instrument
@serializable()
class UserCodeStash(BaseUIDStoreStash):
    object_type = UserCode
    settings: PartitionSettings = PartitionSettings(
        name=UserCode.__canonical_name__, object_type=UserCode
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_all_by_user_verify_key(
        self, credentials: SyftVerifyKey, user_verify_key: SyftVerifyKey
    ) -> Result[List[UserCode], str]:
        qks = QueryKeys(qks=[UserVerifyKeyPartitionKey.with_obj(user_verify_key)])
        return self.query_one(credentials=credentials, qks=qks)

    def get_by_code_hash(
        self, credentials: SyftVerifyKey, code_hash: str
    ) -> Result[Optional[UserCode], str]:
        qks = QueryKeys(qks=[CodeHashPartitionKey.with_obj(code_hash)])
        return self.query_one(credentials=credentials, qks=qks)

    def get_by_service_func_name(
        self, credentials: SyftVerifyKey, service_func_name: str
    ) -> Result[Optional[UserCode], str]:
        qks = QueryKeys(qks=[ServiceFuncNamePartitionKey.with_obj(service_func_name)])
        return self.query_all(
            credentials=credentials, qks=qks, order_by=SubmitTimePartitionKey
        )
