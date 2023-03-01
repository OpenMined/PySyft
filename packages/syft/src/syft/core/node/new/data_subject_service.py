# stdlib
from typing import List
from typing import Optional
from typing import Union

# third party
from result import Result

# relative
from ....telemetry import instrument
from ...common.serde.serializable import serializable
from .context import AuthedServiceContext
from .data_subject import DataSubject
from .data_subject import DataSubjectCreate
from .data_subject import NamePartitionKey
from .document_store import BaseUIDStoreStash
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .document_store import QueryKeys
from .response import SyftError
from .response import SyftSuccess
from .service import AbstractService
from .service import SERVICE_TO_TYPES
from .service import TYPE_TO_SERVICE
from .service import service_method


@instrument
@serializable(recursive_serde=True)
class DataSubjectStash(BaseUIDStoreStash):
    object_type = DataSubject
    settings: PartitionSettings = PartitionSettings(
        name=DataSubject.__canonical_name__, object_type=DataSubject
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(self, name: str) -> Result[Optional[DataSubject], str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(qks=qks)

    def update(self, data_subject: DataSubject) -> Result[DataSubject, str]:
        return self.check_type(data_subject, DataSubject).and_then(super().update)


@instrument
@serializable(recursive_serde=True)
class DataSubjectService(AbstractService):
    store: DocumentStore
    stash: DataSubjectStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = DataSubjectStash(store=store)

    def as_flatten_dict(self, data_subject, flattened_dict):
        members = data_subject.members
        for member in members.values():
            self.as_flatten_dict(member, flattened_dict)

        if data_subject.name not in flattened_dict:
            flattened_dict[data_subject.name] = data_subject

    @service_method(path="data_subject.add", name="add_data_subject")
    def add(
        self, context: AuthedServiceContext, data_subject: DataSubjectCreate
    ) -> Union[SyftSuccess, SyftError]:
        """Register a data subject."""
        flattened_ds_dict = {}
        self.as_flatten_dict(data_subject, flattened_ds_dict)
        registered_count = 0
        for name, data_subject in flattened_ds_dict.items():
            result = self.stash.get_by_name(name=name)
            if result.is_err():
                return SyftError(message=str(result.err()))
            ds_exists = result.ok()

            if ds_exists is None:
                ds = data_subject.to(DataSubject, context=context)
                result = self.stash.set(ds)
                if result.is_err():
                    return SyftError(message=str(result.err()))
                registered_count += 1
        return SyftSuccess(
            message=f"Data Subjects: New Registered: {registered_count}, Existing: {len(flattened_ds_dict)}"
        )

    @service_method(path="data_subject.get_all", name="get_all")
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[List[DataSubject], SyftError]:
        """Get all Data subjects"""
        result = self.stash.get_all()
        if result.is_ok():
            data_subjects = result.ok()
            return data_subjects
        return SyftError(message=result.err())

    @service_method(path="data_subject.get_by_name", name="get_by_name")
    def get_by_name(
        self, context: AuthedServiceContext, name: str
    ) -> Union[SyftSuccess, SyftError]:
        """Get a Data Subject by its name."""
        result = self.stash.get_by_name(name=name)
        if result.is_ok():
            data_subject = result.ok()
            return data_subject
        return SyftError(message=result.err())


TYPE_TO_SERVICE[DataSubject] = DataSubjectService
SERVICE_TO_TYPES[DataSubjectService].update({DataSubject})
