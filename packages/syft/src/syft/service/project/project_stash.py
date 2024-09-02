# stdlib

# third party

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store import UIDPartitionKey
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ...types.uid import UID
from ..request.request import Request
from .project import Project

# TODO: Move to a partitions file?
VerifyKeyPartitionKey = PartitionKey(key="user_verify_key", type_=SyftVerifyKey)
NamePartitionKey = PartitionKey(key="name", type_=str)


@serializable(canonical_name="ProjectStash", version=1)
class ProjectStash(NewBaseUIDStoreStash):
    object_type = Project
    settings: PartitionSettings = PartitionSettings(
        name=Project.__canonical_name__, object_type=Project
    )

    # TODO: Shouldn't this be a list of projects?
    @as_result(StashException)
    def get_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: VerifyKeyPartitionKey
    ) -> list[Request]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        qks = QueryKeys(qks=[VerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_all(
            credentials=credentials,
            qks=qks,
        ).unwrap()

    @as_result(StashException, NotFoundException)
    def get_by_uid(self, credentials: SyftVerifyKey, uid: UID) -> Project:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        return self.query_one(credentials=credentials, qks=qks).unwrap()

    @as_result(StashException, NotFoundException)
    def get_by_name(self, credentials: SyftVerifyKey, project_name: str) -> Project:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(project_name)])
        return self.query_one(credentials=credentials, qks=qks).unwrap()
