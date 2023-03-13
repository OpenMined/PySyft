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
from .project import Project
from .request import Request
from .response import SyftError
from .serializable import serializable

ProjectUserVerifyKeyPartitionKey = PartitionKey(
    key="user_verify_key", type_=SyftVerifyKey
)


@instrument
@serializable(recursive_serde=True)
class ProjectStash(BaseUIDStoreStash):
    object_type = Project
    settings: PartitionSettings = PartitionSettings(
        name=Project.__canonical_name__, object_type=Project
    )

    def get_all_for_verify_key(
        self, verify_key: ProjectUserVerifyKeyPartitionKey
    ) -> Result[List[Request], SyftError]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        qks = QueryKeys(qks=[ProjectUserVerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_all(qks=qks)
