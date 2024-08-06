# stdlib

# third party
from result import Ok
from result import Result
from syft.service.job.job_sql_stash import ObjectStash
from syft.service.user.user_sql import UserDB

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
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
@serializable(canonical_name="UserStashSQL", version=1)
class UserStash(ObjectStash):
    object_type = User
    settings: PartitionSettings = PartitionSettings(
        name=User.__canonical_name__,
        object_type=User,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(server_uid=store.server_uid, ObjectT=User, SchemaT=UserDB)
        self.root_verify_key = store.root_verify_key

    def admin_verify_key(self) -> Result[SyftVerifyKey | None, str]:
        return Ok(self.root_verify_key)

    def admin_user(self) -> Result[User | None, str]:
        return self.get_by_role(
            credentials=self.admin_verify_key().ok(), role=ServiceRole.ADMIN
        )

    def get_by_email(
        self, credentials: SyftVerifyKey, email: str
    ) -> Result[User | None, str]:
        return self.get_one_by_property(
            credentials=credentials, property_name="email", property_value=email
        )

    def email_exists(self, email: str) -> bool:
        res = self.get_by_email(credentials=self.admin_verify_key().ok(), email=email)
        if res.ok() is None:
            return False
        else:
            return True

    def get_by_role(
        self, credentials: SyftVerifyKey, role: ServiceRole
    ) -> Result[User | None, str]:
        return self.get_one_by_property(
            credentials=credentials, property_name="role", property_value=role
        )

    def get_by_signing_key(
        self, credentials: SyftVerifyKey, signing_key: SyftSigningKey
    ) -> Result[User | None, str]:
        return self.get_one_as_admin(
            property_name="signing_key",
            property_value=str(signing_key),
        )

    def get_by_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[User | None, str]:
        return self.get_one_as_admin(
            property_name="verify_key",
            property_value=str(verify_key),
        )
