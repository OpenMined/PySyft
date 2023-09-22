# stdlib
from typing import List
from typing import Optional

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
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
@serializable()
class ProjectStash(BaseUIDStoreStash):
    object_type = Project
    settings: PartitionSettings = PartitionSettings(
        name=Project.__canonical_name__, object_type=Project
    )

    def get_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: VerifyKeyPartitionKey
    ) -> Result[List[Request], SyftError]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        qks = QueryKeys(qks=[VerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_all(
            credentials=credentials,
            qks=qks,
        )

    def get_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[Optional[Project], str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        return self.query_one(credentials=credentials, qks=qks)

    def get_by_name(
        self, credentials: SyftVerifyKey, project_name: str
    ) -> Result[Optional[Project], str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(project_name)])
        return self.query_one(credentials=credentials, qks=qks)
