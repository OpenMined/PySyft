# stdlib

# third party
from result import Result
from result import as_result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store_errors import StashException
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from .data_subject_member import ChildPartitionKey
from .data_subject_member import DataSubjectMemberRelationship
from .data_subject_member import ParentPartitionKey


@instrument
@serializable()
class DataSubjectMemberStash(NewBaseUIDStoreStash):
    object_type = DataSubjectMemberRelationship
    settings: PartitionSettings = PartitionSettings(
        name=DataSubjectMemberRelationship.__canonical_name__,
        object_type=DataSubjectMemberRelationship,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    @as_result(StashException)
    def get_all_for_parent(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[DataSubjectMemberRelationship | None, str]:
        qks = QueryKeys(qks=[ParentPartitionKey.with_obj(name)])
        return self.query_all(credentials=credentials, qks=qks).unwrap()

    @as_result(StashException)
    def get_all_for_child(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[DataSubjectMemberRelationship | None, str]:
        qks = QueryKeys(qks=[ChildPartitionKey.with_obj(name)])
        return self.query_all(credentials=credentials, qks=qks).unwrap()


@instrument
@serializable()
class DataSubjectMemberService(AbstractService):
    store: DocumentStore
    stash: DataSubjectMemberStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = DataSubjectMemberStash(store=store)

    def add(
        self, context: AuthedServiceContext, parent: str, child: str
    ) -> SyftSuccess:
        """Register relationship between data subject and it's member."""
        relation = DataSubjectMemberRelationship(parent=parent, child=child)
        self.stash.set(context.credentials, relation, ignore_duplicates=True).unwrap()
        return SyftSuccess(message=f"Relationship added for: {parent} -> {child}")

    def get_relatives(
        self, context: AuthedServiceContext, data_subject_name: str
    ) -> list[str]:
        """Get all Members for given data subject"""
        result = self.stash.get_all_for_parent(
            context.credentials, name=data_subject_name
        ).unwrap()
        return result


TYPE_TO_SERVICE[DataSubjectMemberRelationship] = DataSubjectMemberService
SERVICE_TO_TYPES[DataSubjectMemberService].update({DataSubjectMemberRelationship})
