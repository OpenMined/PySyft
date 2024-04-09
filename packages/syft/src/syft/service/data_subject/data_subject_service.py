# stdlib

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from .data_subject import DataSubject
from .data_subject import DataSubjectCreate
from .data_subject import NamePartitionKey
from .data_subject_member_service import DataSubjectMemberService


@instrument
@serializable()
class DataSubjectStash(BaseUIDStoreStash):
    object_type = DataSubject
    settings: PartitionSettings = PartitionSettings(
        name=DataSubject.__canonical_name__, object_type=DataSubject
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[DataSubject | None, str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(credentials, qks=qks)

    def update(
        self,
        credentials: SyftVerifyKey,
        data_subject: DataSubject,
        has_permission: bool = False,
    ) -> Result[DataSubject, str]:
        res = self.check_type(data_subject, DataSubject)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().update(credentials=credentials, obj=res.ok())


@instrument
@serializable()
class DataSubjectService(AbstractService):
    store: DocumentStore
    stash: DataSubjectStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = DataSubjectStash(store=store)

    @service_method(path="data_subject.add", name="add_data_subject")
    def add(
        self, context: AuthedServiceContext, data_subject: DataSubjectCreate
    ) -> SyftSuccess | SyftError:
        """Register a data subject."""

        member_relationship_add = context.node.get_service_method(
            DataSubjectMemberService.add
        )

        member_relationships = data_subject.member_relationships
        for member_relationship in member_relationships:
            parent_ds, child_ds = member_relationship
            for ds in [parent_ds, child_ds]:
                result = self.stash.set(
                    context.credentials,
                    ds.to(DataSubject, context=context),
                    ignore_duplicates=True,
                )
                if result.is_err():
                    return SyftError(message=str(result.err()))
            result = member_relationship_add(context, parent_ds.name, child_ds.name)
            if isinstance(result, SyftError):
                return result

        return SyftSuccess(
            message=f"{len(member_relationships)+1} Data Subjects Registered"
        )

    @service_method(path="data_subject.get_all", name="get_all")
    def get_all(self, context: AuthedServiceContext) -> list[DataSubject] | SyftError:
        """Get all Data subjects"""
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            data_subjects = result.ok()
            return data_subjects
        return SyftError(message=result.err())

    @service_method(path="data_subject.get_members", name="members_for")
    def get_members(
        self, context: AuthedServiceContext, data_subject_name: str
    ) -> list[DataSubject] | SyftError:
        get_relatives = context.node.get_service_method(
            DataSubjectMemberService.get_relatives
        )

        relatives = get_relatives(context, data_subject_name)

        if isinstance(relatives, SyftError):
            return relatives

        members = []
        for relative in relatives:
            result = self.get_by_name(context=context, name=relative.child)
            if isinstance(result, SyftError):
                return result
            members.append(result)

        return members

    @service_method(path="data_subject.get_by_name", name="get_by_name")
    def get_by_name(
        self, context: AuthedServiceContext, name: str
    ) -> SyftSuccess | SyftError:
        """Get a Data Subject by its name."""
        result = self.stash.get_by_name(context.credentials, name=name)
        if result.is_ok():
            data_subject = result.ok()
            return data_subject
        return SyftError(message=result.err())


TYPE_TO_SERVICE[DataSubject] = DataSubjectService
SERVICE_TO_TYPES[DataSubjectService].update({DataSubject})
