# stdlib
from typing import Any
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ....telemetry import instrument
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey
from .document_store import BaseStash
from .document_store import CollectionKey
from .document_store import CollectionSettings
from .document_store import DocumentStore
from .document_store import QueryKeys
from .document_store import UIDCollectionKey
from .response import SyftSuccess
from .user import User

# 🟡 TODO 27: it would be nice if these could be defined closer to the User
EmailCollectionKey = CollectionKey(key="email", type_=str)
SigningKeyCollectionKey = CollectionKey(key="signing_key", type_=SyftSigningKey)
VerifyKeyCollectionKey = CollectionKey(key="verify_key", type_=SyftVerifyKey)


@instrument
@serializable(recursive_serde=True)
class UserStash(BaseStash):
    object_type = User
    settings: CollectionSettings = CollectionSettings(
        name=User.__canonical_name__, object_type=User
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def check_type(self, obj: Any, type_: type) -> Result[Any, str]:
        return (
            Ok(obj)
            if isinstance(obj, type_)
            else Err(f"{type(obj)} does not match required type: {type_}")
        )

    def set(self, user: User) -> Result[User, str]:
        return self.check_type(user, self.object_type).and_then(super().set)

    def get_by_uid(self, uid: UID) -> Result[Optional[User], str]:
        qks = QueryKeys(qks=[UIDCollectionKey.with_obj(uid)])
        return self.query_one(qks=qks)

    def get_by_email(self, email: str) -> Result[Optional[User], str]:
        qks = QueryKeys(qks=[EmailCollectionKey.with_obj(email)])
        return self.query_one(qks=qks)

    def get_by_signing_key(
        self, signing_key: SigningKeyCollectionKey
    ) -> Result[Optional[User], str]:
        if isinstance(signing_key, str):
            signing_key = SyftSigningKey.from_string(signing_key)
        qks = QueryKeys(qks=[SigningKeyCollectionKey.with_obj(signing_key)])
        return self.query_one(qks=qks)

    def get_by_verify_key(
        self, verify_key: VerifyKeyCollectionKey
    ) -> Result[Optional[User], str]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        qks = QueryKeys(qks=[VerifyKeyCollectionKey.with_obj(verify_key)])
        return self.query_one(qks=qks)

    def delete_by_uid(self, uid: UID) -> Result[SyftSuccess, str]:
        qk = UIDCollectionKey.with_obj(uid)
        result = super().delete(qk=qk)
        if result.is_ok():
            return Ok(SyftSuccess(message=f"ID: {uid} deleted"))
        return result.err()

    def update(self, user: User) -> Result[User, str]:
        return self.check_type(user, self.object_type).and_then(super().update)
