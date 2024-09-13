# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from .user import User
from .user_roles import ServiceRole


@serializable(canonical_name="UserStashSQL", version=1)
class UserStash(ObjectStash[User]):
    @as_result(StashException, NotFoundException)
    def admin_user(self) -> User:
        # TODO: This returns only one user, the first user with the role ADMIN
        admin_credentials = self.root_verify_key
        return self.get_by_role(
            credentials=admin_credentials, role=ServiceRole.ADMIN
        ).unwrap()

    @as_result(StashException, NotFoundException)
    def get_by_reset_token(self, credentials: SyftVerifyKey, token: str) -> User:
        return self.get_one(
            credentials=credentials,
            filters={"reset_token": token},
        ).unwrap()

    @as_result(StashException, NotFoundException)
    def get_by_email(self, credentials: SyftVerifyKey, email: str) -> User:
        return self.get_one(
            credentials=credentials,
            filters={"email": email},
        ).unwrap()

    @as_result(StashException)
    def email_exists(self, email: str) -> bool:
        try:
            self.get_by_email(credentials=self.root_verify_key, email=email).unwrap()
            return True
        except NotFoundException:
            return False

    @as_result(StashException)
    def verify_key_exists(self, verify_key: SyftVerifyKey) -> bool:
        try:
            self.get_by_verify_key(
                credentials=self.root_verify_key, verify_key=verify_key
            ).unwrap()
            return True
        except NotFoundException:
            return False

    @as_result(StashException, NotFoundException)
    def get_by_role(self, credentials: SyftVerifyKey, role: ServiceRole) -> User:
        try:
            return self.get_one(
                credentials=credentials,
                filters={"role": role},
            ).unwrap()
        except NotFoundException as exc:
            private_msg = f"User with role {role} not found"
            raise NotFoundException.from_exception(exc, private_message=private_msg)

    @as_result(StashException, NotFoundException)
    def get_by_signing_key(
        self, credentials: SyftVerifyKey, signing_key: SyftSigningKey | str
    ) -> User:
        try:
            return self.get_one(
                credentials=credentials,
                filters={"signing_key": signing_key},
            ).unwrap()
        except NotFoundException as exc:
            private_msg = f"User with signing key {signing_key} not found"
            raise NotFoundException.from_exception(exc, private_message=private_msg)

    @as_result(StashException, NotFoundException)
    def get_by_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> User:
        try:
            return self.get_one(
                credentials=credentials,
                filters={"verify_key": verify_key},
            ).unwrap()

        except NotFoundException as exc:
            private_msg = f"User with verify key {verify_key} not found"
            raise NotFoundException.from_exception(exc, private_message=private_msg)
