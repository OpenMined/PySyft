# stdlib

# third party
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...util.telemetry import instrument
from ..request.request import Request
from ..response import SyftError
from .project import Project

VerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
NamePartitionKey = PartitionKey(key="name", type_=str)


@instrument
@serializable(canonical_name="ProjectSQLStash", version=1)
class ProjectStash(ObjectStash[Project]):
    object_type = Project
    settings: PartitionSettings = PartitionSettings(
        name=Project.__canonical_name__, object_type=Project
    )

    def get_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[list[Request], SyftError]:
        return self.get_all_by_field(
            credentials=credentials,
            field_name="user_verify_key",
            field_value=str(verify_key),
        )

    def get_by_name(
        self, credentials: SyftVerifyKey, project_name: str
    ) -> Result[Project | None, str]:
        return self.get_one_by_field(
            credentials=credentials,
            field_name="name",
            field_value=project_name,
        )
