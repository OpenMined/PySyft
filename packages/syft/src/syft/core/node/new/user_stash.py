# stdlib
from typing import Optional

# third party
from result import Ok
from result import Result

# relative
from ....telemetry import instrument
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey
from .document_store import BaseStash
from .document_store import DocumentStore
from .document_store import PartitionKey
from .document_store import PartitionSettings
from .document_store import QueryKeys
from .document_store import UIDPartitionKey
from .response import SyftSuccess
from .serializable import serializable
from .uid import UID
from .user import ServiceRole
from .user import User

# 🟡 TODO 27: it would be nice if these could be defined closer to the User
EmailPartitionKey = PartitionKey(key="email", type_=str)
RolePartitionKey = PartitionKey(key="role", type_=ServiceRole)
SigningKeyPartitionKey = PartitionKey(key="signing_key", type_=SyftSigningKey)
VerifyKeyPartitionKey = PartitionKey(key="verify_key", type_=SyftVerifyKey)


@instrument
@serializable(recursive_serde=True)
class UserStash(BaseStash):
    object_type = User
    settings: PartitionSettings = PartitionSettings(
        name=User.__canonical_name__,
        object_type=User,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set(self, user: User) -> Result[User, str]:
        return self.check_type(user, self.object_type).and_then(super().set)

    def get_by_uid(self, uid: UID) -> Result[Optional[User], str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        return self.query_one(qks=qks)

    def get_by_email(self, email: str) -> Result[Optional[User], str]:
        qks = QueryKeys(qks=[EmailPartitionKey.with_obj(email)])
        return self.query_one(qks=qks)

    def get_by_role(self, role: ServiceRole) -> Result[Optional[User], str]:
        qks = QueryKeys(qks=[RolePartitionKey.with_obj(role)])
        return self.query_one(qks=qks)

    def get_by_signing_key(
        self, signing_key: SyftSigningKey
    ) -> Result[Optional[User], str]:
        if isinstance(signing_key, str):
            signing_key = SyftSigningKey.from_string(signing_key)
        qks = QueryKeys(qks=[SigningKeyPartitionKey.with_obj(signing_key)])
        return self.query_one(qks=qks)

    def get_by_verify_key(
        self, verify_key: SyftVerifyKey
    ) -> Result[Optional[User], str]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        qks = QueryKeys(qks=[VerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_one(qks=qks)

    def delete_by_uid(self, uid: UID) -> Result[SyftSuccess, str]:
        qk = UIDPartitionKey.with_obj(uid)
        result = super().delete(qk=qk)
        if result.is_ok():
            return Ok(SyftSuccess(message=f"ID: {uid} deleted"))
        return result

    def update(self, user: User) -> Result[User, str]:
        return self.check_type(user, self.object_type).and_then(super().update)
