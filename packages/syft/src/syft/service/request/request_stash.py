# stdlib

# third party

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...types.datetime import DateTime
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.uid import UID
from .request import Request

RequestingUserVerifyKeyPartitionKey = PartitionKey(
    key="requesting_user_verify_key", type_=SyftVerifyKey
)

OrderByRequestTimeStampPartitionKey = PartitionKey(key="request_time", type_=DateTime)


@serializable(canonical_name="RequestStash", version=1)
class RequestStash(NewBaseUIDStoreStash):
    object_type = Request
    settings: PartitionSettings = PartitionSettings(
        name=Request.__canonical_name__, object_type=Request
    )

    @as_result(SyftException)
    def get_all_for_verify_key(
        self,
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
    ) -> list[Request]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        qks = QueryKeys(qks=[RequestingUserVerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_all(
            credentials=credentials,
            qks=qks,
            order_by=OrderByRequestTimeStampPartitionKey,
        ).unwrap()

    @as_result(SyftException)
    def get_by_usercode_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> list[Request]:
        all_requests = self.get_all(credentials=credentials).unwrap()
        res = []
        for r in all_requests:
            try:
                if r.code_id == user_code_id:
                    res.append(r)
            except SyftException:
                pass
        return res
