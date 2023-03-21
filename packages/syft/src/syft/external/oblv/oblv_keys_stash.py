# stdlib
from typing import Any
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...core.node.new.document_store import BaseStash
from ...core.node.new.document_store import DocumentStore
from ...core.node.new.document_store import PartitionSettings
from ...core.node.new.document_store import QueryKeys
from ...core.node.new.document_store import UIDPartitionKey
from ...core.node.new.serializable import serializable
from ...core.node.new.uid import UID
from .oblv_keys import OblvKeys


@serializable(recursive_serde=True)
class OblvKeysStash(BaseStash):
    object_type = OblvKeys
    settings: PartitionSettings = PartitionSettings(
        name=OblvKeys.__canonical_name__, object_type=OblvKeys, db_name="app"
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def check_type(self, obj: Any, type_: type) -> Result[Any, str]:
        return (
            Ok(obj)
            if isinstance(obj, type_)
            else Err(f"{type(obj)} does not match required type: {type_}")
        )

    def set(self, oblv_keys: OblvKeys) -> Result[OblvKeys, Err]:
        if not len(self):
            return self.check_type(oblv_keys, self.object_type).and_then(super().set)
        else:
            return Err("Domain Node already has an existing public/private key pair")

    def get_by_uid(self, uid: UID) -> Result[Optional[OblvKeys], str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        return Ok(self.query_one(qks=qks))

    def delete_by_uid(self, uid: UID) -> Result[bool, str]:
        qk = UIDPartitionKey.with_obj(uid)
        return super().delete(qk=qk)

    def update(self, task: OblvKeys) -> Result[OblvKeys, str]:
        return self.check_type(task, self.object_type).and_then(super().update)
