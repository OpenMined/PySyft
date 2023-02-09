# stdlib
from typing import Any
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ....common.serde.serializable import serializable
from ..document_store import BaseStash
from ..document_store import CollectionKey
from ..document_store import CollectionSettings
from ..document_store import DocumentStore
from ..document_store import QueryKeys
from .oblv_keys import OblvKeys

IntegerIDCollectionKey = CollectionKey(key="id_int", type_=int)


@serializable(recursive_serde=True)
class OblvKeysStash(BaseStash):
    object_type = OblvKeys
    settings: CollectionSettings = CollectionSettings(
        name=OblvKeys.__canonical_name__, object_type=OblvKeys
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

        row_exists = self.get_by_id(id_int=1).ok()

        if row_exists.is_ok() and row_exists.ok():
            return Err("Domain Node already has an existing public/private key pair")
        else:
            # Temporary fix to ensure only one item in Database since we do not have .all()
            oblv_keys.id_int = 1

        return self.check_type(oblv_keys, self.object_type).and_then(super().set)

    def get_by_id(self, id_int: int) -> Result[Optional[OblvKeys], str]:
        qks = QueryKeys(qks=[IntegerIDCollectionKey.with_obj(id_int)])
        return Ok(self.query_one(qks=qks))

    def delete_by_id(self, id_int: int) -> Result[bool, str]:
        qk = IntegerIDCollectionKey.with_obj(id_int)
        return super().delete(qk=qk)

    def update(self, task: OblvKeys) -> Result[OblvKeys, str]:
        return self.check_type(task, self.object_type).and_then(super().update)
