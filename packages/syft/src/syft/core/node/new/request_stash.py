# stdlib
from typing import List

# third party
from result import Result

# relative
from ....telemetry import instrument
from .credentials import SyftVerifyKey
from .document_store import BaseUIDStoreStash
from .document_store import PartitionKey
from .document_store import PartitionSettings
from .document_store import QueryKeys
from .request import Request
from .request import RequestStatus
from .response import SyftError
from .serializable import serializable

RequestingUserVerifyKeyPartitionKey = PartitionKey(
    key="requesting_user_verify_key", type_=SyftVerifyKey
)
StatusPartitionKey = PartitionKey(key="status", type_=RequestStatus)


@instrument
@serializable(recursive_serde=True)
class RequestStash(BaseUIDStoreStash):
    object_type = Request
    settings: PartitionSettings = PartitionSettings(
        name=Request.__canonical_name__, object_type=Request
    )

    def get_all_for_verify_key(
        self, verify_key: RequestingUserVerifyKeyPartitionKey
    ) -> Result[List[Request], SyftError]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        qks = QueryKeys(qks=[RequestingUserVerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_all(qks=qks)

    def get_all_for_status(
        self, status: RequestStatus
    ) -> Result[List[Request], SyftError]:
        qks = QueryKeys(qks=[StatusPartitionKey.with_obj(status)])
        return self.query_all(qks=qks)
