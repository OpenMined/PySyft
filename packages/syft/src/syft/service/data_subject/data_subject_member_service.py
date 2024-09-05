# stdlib

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from .data_subject_member import ChildPartitionKey
from .data_subject_member import DataSubjectMemberRelationship
from .data_subject_member import ParentPartitionKey


@serializable(canonical_name="DataSubjectMemberStash", version=1)
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
    ) -> list[DataSubjectMemberRelationship]:
        qks = QueryKeys(qks=[ParentPartitionKey.with_obj(name)])
        return self.query_all(credentials=credentials, qks=qks).unwrap()

    @as_result(StashException)
    def get_all_for_child(
        self, credentials: SyftVerifyKey, name: str
    ) -> list[DataSubjectMemberRelationship]:
        qks = QueryKeys(qks=[ChildPartitionKey.with_obj(name)])
        return self.query_all(credentials=credentials, qks=qks).unwrap()


@serializable(canonical_name="DataSubjectMemberService", version=1)
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
    ) -> list[DataSubjectMemberRelationship]:
        """Get all Members for given data subject"""
        return self.stash.get_all_for_parent(
            context.credentials, name=data_subject_name
        ).unwrap()


TYPE_TO_SERVICE[DataSubjectMemberRelationship] = DataSubjectMemberService
SERVICE_TO_TYPES[DataSubjectMemberService].update({DataSubjectMemberRelationship})
