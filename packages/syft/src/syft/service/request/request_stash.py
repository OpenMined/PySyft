# stdlib

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...types.datetime import DateTime
from ...util.telemetry import instrument
from .request import Request

RequestingUserVerifyKeyPartitionKey = PartitionKey(
    key="requesting_user_verify_key", type_=SyftVerifyKey
)

OrderByRequestTimeStampPartitionKey = PartitionKey(key="request_time", type_=DateTime)


@instrument
@serializable()
class RequestStash(BaseUIDStoreStash):
    object_type = Request
    settings: PartitionSettings = PartitionSettings(
        name=Request.__canonical_name__, object_type=Request
    )

    def get_all_for_verify_key(
        self,
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
    ) -> Result[list[Request], str]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        qks = QueryKeys(qks=[RequestingUserVerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_all(
            credentials=credentials,
            qks=qks,
            order_by=OrderByRequestTimeStampPartitionKey,
        )
