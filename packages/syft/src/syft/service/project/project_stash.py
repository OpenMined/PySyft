# stdlib
from typing import List

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...util.telemetry import instrument
from ..request.request import Request
from ..response import SyftError
from .project import Project

ProjectUserVerifyKeyPartitionKey = PartitionKey(
    key="user_verify_key", type_=SyftVerifyKey
)


@instrument
@serializable()
class ProjectStash(BaseUIDStoreStash):
    object_type = Project
    settings: PartitionSettings = PartitionSettings(
        name=Project.__canonical_name__, object_type=Project
    )

    def get_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: ProjectUserVerifyKeyPartitionKey
    ) -> Result[List[Request], SyftError]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        qks = QueryKeys(qks=[ProjectUserVerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_all(credentials=credentials, qks=qks)
