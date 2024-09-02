# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store import UIDPartitionKey
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ...types.uid import UID
from .user import User
from .user_roles import ServiceRole

# 🟡 TODO 27: it would be nice if these could be defined closer to the User
EmailPartitionKey = PartitionKey(key="email", type_=str)
PasswordResetTokenPartitionKey = PartitionKey(key="reset_token", type_=str)
RolePartitionKey = PartitionKey(key="role", type_=ServiceRole)
SigningKeyPartitionKey = PartitionKey(key="signing_key", type_=SyftSigningKey)
VerifyKeyPartitionKey = PartitionKey(key="verify_key", type_=SyftVerifyKey)


@serializable(canonical_name="UserStash", version=1)
class UserStash(NewBaseUIDStoreStash):
    object_type = User
    settings: PartitionSettings = PartitionSettings(
        name=User.__canonical_name__,
        object_type=User,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def admin_verify_key(self) -> SyftVerifyKey:
        return self.partition.root_verify_key

    @as_result(StashException, NotFoundException)
    def admin_user(self) -> User:
        # TODO: This returns only one user, the first user with the role ADMIN
        admin_credentials = self.admin_verify_key()
        return self.get_by_role(
            credentials=admin_credentials, role=ServiceRole.ADMIN
        ).unwrap()

    @as_result(StashException, NotFoundException)
    def get_by_uid(self, credentials: SyftVerifyKey, uid: UID) -> User:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        try:
            return self.query_one(credentials=credentials, qks=qks).unwrap()
        except NotFoundException as exc:
            raise NotFoundException.from_exception(
                exc, public_message=f"User {uid} not found"
            )

    @as_result(StashException, NotFoundException)
    def get_by_reset_token(self, credentials: SyftVerifyKey, token: str) -> User:
        qks = QueryKeys(qks=[PasswordResetTokenPartitionKey.with_obj(token)])
        return self.query_one(credentials=credentials, qks=qks).unwrap()

    @as_result(StashException, NotFoundException)
    def get_by_email(self, credentials: SyftVerifyKey, email: str) -> User:
        qks = QueryKeys(qks=[EmailPartitionKey.with_obj(email)])

        try:
            return self.query_one(credentials=credentials, qks=qks).unwrap()
        except NotFoundException as exc:
            raise NotFoundException.from_exception(
                exc, public_message=f"User {email} not found"
            )

    @as_result(StashException)
    def email_exists(self, email: str) -> bool:
        try:
            self.get_by_email(credentials=self.admin_verify_key(), email=email).unwrap()
            return True
        except NotFoundException:
            return False

    @as_result(StashException, NotFoundException)
    def get_by_role(self, credentials: SyftVerifyKey, role: ServiceRole) -> User:
        # TODO: Is this method correct? Should'nt it return a list of all member with a particular role?
        qks = QueryKeys(qks=[RolePartitionKey.with_obj(role)])

        try:
            return self.query_one(credentials=credentials, qks=qks).unwrap()
        except NotFoundException as exc:
            private_msg = f"User with role {role} not found"
            raise NotFoundException.from_exception(exc, private_message=private_msg)

    @as_result(StashException, NotFoundException)
    def get_by_signing_key(
        self, credentials: SyftVerifyKey, signing_key: SyftSigningKey | str
    ) -> User:
        if isinstance(signing_key, str):
            signing_key = SyftSigningKey.from_string(signing_key)

        qks = QueryKeys(qks=[SigningKeyPartitionKey.with_obj(signing_key)])

        try:
            return self.query_one(credentials=credentials, qks=qks).unwrap()
        except NotFoundException as exc:
            private_msg = f"User with signing key {signing_key} not found"
            raise NotFoundException.from_exception(exc, private_message=private_msg)

    @as_result(StashException, NotFoundException)
    def get_by_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey | str
    ) -> User:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)

        qks = QueryKeys(qks=[VerifyKeyPartitionKey.with_obj(verify_key)])

        try:
            return self.query_one(credentials=credentials, qks=qks).unwrap()
        except NotFoundException as exc:
            private_msg = f"User with verify key {verify_key} not found"
            raise NotFoundException.from_exception(exc, private_message=private_msg)
