# stdlib
from typing import Any
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ..document_store import BaseStash
from ..document_store import CollectionSettings
from ..document_store import DocumentStore
from ..document_store import QueryKeys
from ..document_store import UIDCollectionKey
from .task import Task


@serializable(recursive_serde=True)
class TaskStash(BaseStash):
    object_type = Task
    settings: CollectionSettings = CollectionSettings(
        name=Task.__canonical_name__, object_type=Task
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def check_type(self, obj: Any, type_: type) -> Result[Any, str]:
        return (
            Ok(obj)
            if isinstance(obj, type_)
            else Err(f"{type(obj)} does not match required type: {type_}")
        )

    def set(self, task: Task) -> Result[Task, str]:
        return self.check_type(task, self.object_type).and_then(super().set)

    def get_by_uid(self, uid: UID) -> Result[Optional[Task], str]:
        qks = QueryKeys(qks=[UIDCollectionKey.with_obj(uid)])
        return Ok(self.query_one(qks=qks))

    def delete_by_uid(self, uid: UID) -> Result[bool, str]:
        qk = UIDCollectionKey.with_obj(uid)
        return super().delete(qk=qk)

    def update(self, task: Task) -> Result[Task, str]:
        return self.check_type(task, self.object_type).and_then(super().update)
