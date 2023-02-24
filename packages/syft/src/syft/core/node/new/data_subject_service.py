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
from .data_subject import NamePartitionKey
from .dataset import Dataset
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

    def update(self, data_subject: DataSubject) -> Result[Dataset, str]:
        return self.check_type(data_subject, DataSubject).and_then(super().update)


@instrument
@serializable(recursive_serde=True)
class DataSubjectService(AbstractService):
    store: DocumentStore
    stash: DataSubjectStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = DataSubjectStash(store=store)

    @service_method(path="data_subject.add", name="add_data_subject")
    def add(
        self, context: AuthedServiceContext, data_subject: DataSubject
    ) -> Union[SyftSuccess, SyftError]:
        """Add a Dataset"""
        data_subject_members = list(data_subject.members.values())
        data_subjects = [data_subject] + data_subject_members
        for data_subject in data_subjects:
            result = self.stash.set(data_subject, ignore_duplicates=True)
            if result.is_err():
                return SyftError(message=str(result.err()))
        return SyftSuccess(message="Data Subject Added")

    @service_method(path="data_subject.get_all", name="get_all")
    def get_all(self, context: AuthedServiceContext) -> Union[List[Dataset], SyftError]:
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
        """Get a Dataset subject"""
        result = self.stash.get_by_name(name=name)
        if result.is_ok():
            data_subject = result.ok()
            return data_subject
        return SyftError(message=result.err())


TYPE_TO_SERVICE[DataSubject] = DataSubjectService
SERVICE_TO_TYPES[DataSubjectService].update({DataSubject})
