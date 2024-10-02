# stdlib

# third party

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.db import DBManager
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from .data_subject import DataSubject
from .data_subject import DataSubjectCreate


@serializable(canonical_name="DataSubjectSQLStash", version=1)
class DataSubjectStash(ObjectStash[DataSubject]):
    @as_result(StashException)
    def get_by_name(self, credentials: SyftVerifyKey, name: str) -> DataSubject:
        return self.get_one(
            credentials=credentials,
            filters={"name": name},
        ).unwrap()


@serializable(canonical_name="DataSubjectService", version=1)
class DataSubjectService(AbstractService):
    stash: DataSubjectStash

    def __init__(self, store: DBManager) -> None:
        self.stash = DataSubjectStash(store=store)

    @service_method(path="data_subject.add", name="add_data_subject")
    def add(
        self, context: AuthedServiceContext, data_subject: DataSubjectCreate
    ) -> SyftSuccess:
        """Register a data subject."""  #

        member_relationships: set[tuple[str, str]] = data_subject.member_relationships
        if len(member_relationships) == 0:
            self.stash.set(
                context.credentials,
                data_subject.to(DataSubject, context=context),
            ).unwrap()
        else:
            for member_relationship in member_relationships:
                parent_ds, child_ds = member_relationship
                for ds in [parent_ds, child_ds]:
                    self.stash.set(
                        context.credentials,
                        ds.to(DataSubject, context=context),
                        ignore_duplicates=True,
                    ).unwrap()
                context.server.services.data_subject_member.add(
                    context, parent_ds.name, child_ds.name
                )

        return SyftSuccess(
            message=f"{len(member_relationships)+1} Data Subjects Registered",
            value=member_relationships,
        )

    @service_method(path="data_subject.get_all", name="get_all")
    def get_all(self, context: AuthedServiceContext) -> list[DataSubject]:
        """Get all Data subjects"""
        return self.stash.get_all(context.credentials).unwrap()

    @service_method(path="data_subject.get_members", name="members_for")
    def get_members(
        self, context: AuthedServiceContext, data_subject_name: str
    ) -> list[DataSubject]:
        relatives = context.server.services.data_subject.get_relatives(
            context, data_subject_name
        )

        members = []
        for relative in relatives:
            result = self.get_by_name(context=context, name=relative.child)
            members.append(result)

        return members

    @service_method(path="data_subject.get_by_name", name="get_by_name")
    def get_by_name(self, context: AuthedServiceContext, name: str) -> DataSubject:
        """Get a Data Subject by its name."""
        return self.stash.get_by_name(context.credentials, name=name).unwrap()


TYPE_TO_SERVICE[DataSubject] = DataSubjectService
SERVICE_TO_TYPES[DataSubjectService].update({DataSubject})
