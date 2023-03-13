# stdlib
from typing import List
from typing import Optional
from typing import Union

# third party
from result import Result

# relative
from ....telemetry import instrument
from .context import AuthedServiceContext
from .data_subject_member import ChildPartitionKey
from .data_subject_member import DataSubjectMemberRelationship
from .data_subject_member import ParentPartitionKey
from .document_store import BaseUIDStoreStash
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .document_store import QueryKeys
from .response import SyftError
from .response import SyftSuccess
from .serializable import serializable
from .service import AbstractService
from .service import SERVICE_TO_TYPES
from .service import TYPE_TO_SERVICE


@instrument
@serializable(recursive_serde=True)
class DataSubjectMemberStash(BaseUIDStoreStash):
    object_type = DataSubjectMemberRelationship
    settings: PartitionSettings = PartitionSettings(
        name=DataSubjectMemberRelationship.__canonical_name__,
        object_type=DataSubjectMemberRelationship,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_all_for_parent(
        self, name: str
    ) -> Result[Optional[DataSubjectMemberRelationship], str]:
        qks = QueryKeys(qks=[ParentPartitionKey.with_obj(name)])
        return self.query_all(qks=qks)

    def get_all_for_child(
        self, name: str
    ) -> Result[Optional[DataSubjectMemberRelationship], str]:
        qks = QueryKeys(qks=[ChildPartitionKey.with_obj(name)])
        return self.query_all(qks=qks)


@instrument
@serializable(recursive_serde=True)
class DataSubjectMemberService(AbstractService):
    store: DocumentStore
    stash: DataSubjectMemberStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = DataSubjectMemberStash(store=store)

    def add(
        self, context: AuthedServiceContext, parent: str, child: str
    ) -> Union[SyftSuccess, SyftError]:
        """Register relationship between data subject and it's member."""
        relation = DataSubjectMemberRelationship(parent=parent, child=child)
        result = self.stash.set(relation, ignore_duplicates=True)
        if result.is_err():
            return SyftError(result.err())
        return SyftSuccess(message=f"Relationship added for: {parent} -> {child}")

    def get_relatives(
        self, context: AuthedServiceContext, data_subject_name: str
    ) -> Union[List[str], SyftError]:
        """Get all Members for given data subject"""
        result = self.stash.get_all_for_parent(name=data_subject_name)
        if result.is_ok():
            data_subject_members = result.ok()
            return data_subject_members
        return SyftError(message=result.err())


TYPE_TO_SERVICE[DataSubjectMemberRelationship] = DataSubjectMemberService
SERVICE_TO_TYPES[DataSubjectMemberService].update({DataSubjectMemberRelationship})
