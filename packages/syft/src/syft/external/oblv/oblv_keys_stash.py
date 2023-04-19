# stdlib
from typing import Any
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store import UIDPartitionKey
from ...types.uid import UID
from .oblv_keys import OblvKeys


@serializable()
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

    def set(
        self, credentials: SyftVerifyKey, oblv_keys: OblvKeys
    ) -> Result[OblvKeys, Err]:
        if not len(self):
            valid = self.check_type(oblv_keys, self.object_type)
            if valid.is_err():
                return SyftError(message=valid.err())

            return super().set(credentials, oblv_keys)
        else:
            return Err("Domain Node already has an existing public/private key pair")

    def get_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[Optional[OblvKeys], str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        return Ok(self.query_one(credentials=credentials, qks=qks))

    def delete_by_uid(self, credentials: SyftVerifyKey, uid: UID) -> Result[bool, str]:
        qk = UIDPartitionKey.with_obj(uid)
        return super().delete(qk=qk)

    def update(
        self, credentials: SyftVerifyKey, task: OblvKeys
    ) -> Result[OblvKeys, str]:
        valid = self.check_type(task, self.object_type)
        if valid.is_err():
            return SyftError(message=valid.err())

        return super().update(credentials, task)
