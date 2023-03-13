# stdlib
from typing import List
from typing import Optional

# third party
from result import Result

# relative
from ....telemetry import instrument
from .credentials import SyftVerifyKey
from .document_store import BaseUIDStoreStash
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .document_store import QueryKeys
from .serializable import serializable
from .user_code import CodeHashPartitionKey
from .user_code import UserCode
from .user_code import UserVerifyKeyPartitionKey


@instrument
@serializable(recursive_serde=True)
class UserCodeStash(BaseUIDStoreStash):
    object_type = UserCode
    settings: PartitionSettings = PartitionSettings(
        name=UserCode.__canonical_name__, object_type=UserCode
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_all_by_user_verify_key(
        self, user_verify_key: SyftVerifyKey
    ) -> Result[List[UserCode], str]:
        qks = QueryKeys(qks=[UserVerifyKeyPartitionKey.with_obj(user_verify_key)])
        return self.query_one(qks=qks)

    def get_by_code_hash(self, code_hash: int) -> Result[Optional[UserCode], str]:
        qks = QueryKeys(qks=[CodeHashPartitionKey.with_obj(code_hash)])
        return self.query_one(qks=qks)
