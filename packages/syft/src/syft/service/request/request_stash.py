# stdlib

# third party
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.datetime import DateTime
from ...types.uid import UID
from ...util.telemetry import instrument
from .request import Request

RequestingUserVerifyKeyPartitionKey = PartitionKey(
    key="requesting_user_verify_key", type_=SyftVerifyKey
)

OrderByRequestTimeStampPartitionKey = PartitionKey(key="request_time", type_=DateTime)


@instrument
@serializable(canonical_name="RequestStashSQL", version=1)
class RequestStash(ObjectStash[Request]):
    settings: PartitionSettings = PartitionSettings(
        name=Request.__canonical_name__, object_type=Request
    )

    def get_all_for_verify_key(
        self,
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
    ) -> Result[list[Request], str]:
        return self.get_all_by_field(
            credentials=credentials,
            field_name="requesting_user_verify_key",
            field_value=str(verify_key),
        )

    def get_by_usercode_id(
        self, credentials: SyftVerifyKey, user_code_id: UID
    ) -> Result[list[Request], str]:
        return self.get_all_by_field(
            credentials=credentials,
            field_name="code_id",
            field_value=str(user_code_id),
        )
