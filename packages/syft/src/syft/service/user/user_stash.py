# stdlib

# third party
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...store.db.base_stash import ObjectStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...util.telemetry import instrument
from .user import User
from .user_roles import ServiceRole

# 🟡 TODO 27: it would be nice if these could be defined closer to the User
EmailPartitionKey = PartitionKey(key="email", type_=str)
PasswordResetTokenPartitionKey = PartitionKey(key="reset_token", type_=str)
RolePartitionKey = PartitionKey(key="role", type_=ServiceRole)
SigningKeyPartitionKey = PartitionKey(key="signing_key", type_=SyftSigningKey)
VerifyKeyPartitionKey = PartitionKey(key="verify_key", type_=SyftVerifyKey)


@instrument
@serializable(canonical_name="UserStashSQL", version=1)
class UserStash(ObjectStash[User]):
    object_type = User
    settings: PartitionSettings = PartitionSettings(
        name=User.__canonical_name__,
        object_type=User,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

        self._init_root()

    def _init_root(self) -> None:
        # start a transaction
        users = self.get_all(self.root_verify_key, has_permission=True)
        if not users:
            # NOTE this is not thread safe, should use a session and transaction
            super().set(
                self.root_verify_key,
                User(
                    email="_internal@root.com",
                    role=ServiceRole.ADMIN,
                    verify_key=self.root_verify_key,
                ),
            )

    def admin_verify_key(self) -> Result[SyftVerifyKey | None, str]:
        return Ok(self.root_verify_key)

    def admin_user(self) -> Result[User | None, str]:
        return self.get_by_role(
            credentials=self.admin_verify_key().ok(), role=ServiceRole.ADMIN
        )

    def get_by_reset_token(
        self, credentials: SyftVerifyKey, token: str
    ) -> Result[User | None, str]:
        return self.get_one_by_field(
            credentials=credentials, field_name="reset_token", field_value=token
        )

    def get_by_email(
        self, credentials: SyftVerifyKey, email: str
    ) -> Result[User | None, str]:
        return self.get_one_by_field(
            credentials=credentials, field_name="email", field_value=email
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
        return self.get_one_by_field(
            credentials=credentials, field_name="role", field_value=role
        )

    def get_by_signing_key(
        self, credentials: SyftVerifyKey, signing_key: SyftSigningKey
    ) -> Result[User | None, str]:
        return self.get_one_by_field(
            credentials=credentials,
            field_name="signing_key",
            field_value=str(signing_key),
        )

    def get_by_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[User | None, str]:
        return self.get_one_by_field(
            credentials=credentials,
            field_name="verify_key",
            field_value=str(verify_key),
        )
