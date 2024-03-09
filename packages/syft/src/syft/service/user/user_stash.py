# stdlib
from typing import List
from typing import Optional

# third party
from result import Ok
from result import Result

# relative
from ...node.credentials import SyftSigningKey
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store import UIDPartitionKey
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from ..response import SyftSuccess
from .user import User
from .user_roles import ServiceRole

# 🟡 TODO 27: it would be nice if these could be defined closer to the User
EmailPartitionKey = PartitionKey(key="email", type_=str)
RolePartitionKey = PartitionKey(key="role", type_=ServiceRole)
SigningKeyPartitionKey = PartitionKey(key="signing_key", type_=SyftSigningKey)
VerifyKeyPartitionKey = PartitionKey(key="verify_key", type_=SyftVerifyKey)


@instrument
@serializable()
class UserStash(BaseStash):
    object_type = User
    settings: PartitionSettings = PartitionSettings(
        name=User.__canonical_name__,
        object_type=User,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set(
        self,
        credentials: SyftVerifyKey,
        user: User,
        add_permissions: Optional[List[ActionObjectPermission]] = None,
        ignore_duplicates: bool = False,
    ) -> Result[User, str]:
        res = self.check_type(user, self.object_type)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().set(
            credentials=credentials,
            obj=res.ok(),
            add_permissions=add_permissions,
            ignore_duplicates=ignore_duplicates,
        )

    def admin_verify_key(self) -> Result[Optional[SyftVerifyKey], str]:
        return Ok(self.partition.root_verify_key)

    def admin_user(self) -> Result[Optional[User], str]:
        return self.get_by_role(
            credentials=self.admin_verify_key().ok(), role=ServiceRole.ADMIN
        )

    def get_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[Optional[User], str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        return self.query_one(credentials=credentials, qks=qks)

    def get_by_email(
        self, credentials: SyftVerifyKey, email: str
    ) -> Result[Optional[User], str]:
        qks = QueryKeys(qks=[EmailPartitionKey.with_obj(email)])
        return self.query_one(credentials=credentials, qks=qks)

    def email_exists(self, email: str) -> bool:
        res = self.get_by_email(credentials=self.admin_verify_key().ok(), email=email)
        if res.ok() is None:
            return False
        else:
            return True

    def get_by_role(
        self, credentials: SyftVerifyKey, role: ServiceRole
    ) -> Result[Optional[User], str]:
        qks = QueryKeys(qks=[RolePartitionKey.with_obj(role)])
        return self.query_one(credentials=credentials, qks=qks)

    def get_by_signing_key(
        self, credentials: SyftVerifyKey, signing_key: SyftSigningKey
    ) -> Result[Optional[User], str]:
        if isinstance(signing_key, str):
            signing_key = SyftSigningKey.from_string(signing_key)
        qks = QueryKeys(qks=[SigningKeyPartitionKey.with_obj(signing_key)])
        return self.query_one(credentials=credentials, qks=qks)

    def get_by_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[Optional[User], str]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        qks = QueryKeys(qks=[VerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_one(credentials=credentials, qks=qks)

    def delete_by_uid(
        self, credentials: SyftVerifyKey, uid: UID, has_permission: bool = False
    ) -> Result[SyftSuccess, str]:
        qk = UIDPartitionKey.with_obj(uid)
        result = super().delete(
            credentials=credentials, qk=qk, has_permission=has_permission
        )
        if result.is_ok():
            return Ok(SyftSuccess(message=f"ID: {uid} deleted"))
        return result

    def update(
        self, credentials: SyftVerifyKey, user: User, has_permission: bool = False
    ) -> Result[User, str]:
        res = self.check_type(user, self.object_type)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().update(
            credentials=credentials, obj=res.ok(), has_permission=has_permission
        )
