# stdlib

# third party
from git import Object
from result import Result
from syft.service.job.job_sql_stash import ObjectStash
from syft.service.project.project_sql import ProjectDB

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store import UIDPartitionKey
from ...types.uid import UID
from ...util.telemetry import instrument
from ..request.request import Request
from ..response import SyftError
from .project import Project

VerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
NamePartitionKey = PartitionKey(key="name", type_=str)


@instrument
@serializable(canonical_name="ProjectStashSQL", version=1)
class ProjectStashSQL(ObjectStash[Project, ProjectDB]):
    object_type = Project
    settings: PartitionSettings = PartitionSettings(
        name=Project.__canonical_name__, object_type=Project
    )

    def get_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[list[Request], SyftError]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        return self.get_many_by_property(
            credentials=credentials,
            property_name="user_verify_key",
            property_value=verify_key,
        )

    def get_by_name(
        self, credentials: SyftVerifyKey, project_name: str
    ) -> Result[Project | None, str]:
        return self.get_one_by_property(
            credentials=credentials,
            property_name="name",
            property_value=project_name,
        )
