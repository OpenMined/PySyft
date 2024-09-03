# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ...types.uid import UID
from ...util.trace_decorator import instrument
from .user import User
from .user_roles import ServiceRole


@instrument
@serializable(canonical_name="UserStashSQL", version=1)
class UserStash(ObjectStash[User]):
    @as_result(StashException)
    def init_root_user(self) -> None:
        # start a transaction
        users = self.get_all(self.root_verify_key, has_permission=True).unwrap()
        if not users:
            # NOTE this is not thread safe, should use a session and transaction
            super().set(
                self.root_verify_key,
                User(
                    id=UID(),
                    email="_internal@root.com",
                    role=ServiceRole.ADMIN,
                    verify_key=self.root_verify_key,
                ),
            )

    def admin_verify_key(self) -> SyftVerifyKey:
        return self.root_verify_key

    @as_result(StashException, NotFoundException)
    def admin_user(self) -> User:
        # TODO: This returns only one user, the first user with the role ADMIN
        admin_credentials = self.admin_verify_key()
        return self.get_by_role(
            credentials=admin_credentials, role=ServiceRole.ADMIN
        ).unwrap()

    @as_result(StashException, NotFoundException)
    def get_by_reset_token(self, credentials: SyftVerifyKey, token: str) -> User:
        return self.get_one_by_field(
            credentials=credentials, field_name="reset_token", field_value=token
        ).unwrap()

    @as_result(StashException, NotFoundException)
    def get_by_email(self, credentials: SyftVerifyKey, email: str) -> User:
        return self.get_one_by_field(
            credentials=credentials, field_name="email", field_value=email
        ).unwrap()

    @as_result(StashException)
    def email_exists(self, email: str) -> bool:
        try:
            self.get_by_email(credentials=self.admin_verify_key(), email=email).unwrap()
            return True
        except NotFoundException:
            return False

    @as_result(StashException, NotFoundException)
    def get_by_role(self, credentials: SyftVerifyKey, role: ServiceRole) -> User:
        try:
            return self.get_one_by_field(
                credentials=credentials, field_name="role", field_value=role
            ).unwrap()
        except NotFoundException as exc:
            private_msg = f"User with role {role} not found"
            raise NotFoundException.from_exception(exc, private_message=private_msg)

    @as_result(StashException, NotFoundException)
    def get_by_signing_key(
        self, credentials: SyftVerifyKey, signing_key: SyftSigningKey | str
    ) -> User:
        try:
            return self.get_one_by_field(
                credentials=credentials,
                field_name="signing_key",
                field_value=str(signing_key),
            ).unwrap()
        except NotFoundException as exc:
            private_msg = f"User with signing key {signing_key} not found"
            raise NotFoundException.from_exception(exc, private_message=private_msg)

    @as_result(StashException, NotFoundException)
    def get_by_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> User:
        try:
            return self.get_one_by_field(
                credentials=credentials,
                field_name="verify_key",
                field_value=str(verify_key),
            ).unwrap()
        except NotFoundException as exc:
            private_msg = f"User with verify key {verify_key} not found"
            raise NotFoundException.from_exception(exc, private_message=private_msg)
