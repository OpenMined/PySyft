# stdlib

# stdlib
from typing import Any
from typing import cast

# third party
from result import Err
from result import Ok

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
from ...store.errors import NotFoundError
from ...store.errors import StashError
from ...types.result import as_result
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from .errors import UserCreateError
from .errors import UserDeleteError
from .errors import UserUpdateError
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

    @as_result(StashError, NotFoundError)
    def _query_one(self, credentials: SyftVerifyKey, qks: QueryKeys) -> User:
        result = super().query_one(credentials=credentials, qks=qks)

        match result:
            case Ok(user):
                return cast(User, user)
            case Ok(None):
                raise NotFoundError
            case Err(msg):
                raise StashError(msg)
            case _:
                raise StashError

    def _admin_verify_key(self) -> SyftVerifyKey:
        return self.partition.root_verify_key

    @as_result(StashError, UserCreateError)
    def _set(
        self,
        credentials: SyftVerifyKey,
        user: User,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> User:
        _user = self._check_type(user, self.object_type).unwrap()

        result = super().set(
            credentials=credentials,
            obj=_user,
            add_permissions=add_permissions,
            ignore_duplicates=ignore_duplicates,
            add_storage_permission=add_storage_permission,
        )

        match result:
            case Ok(new_user):
                return cast(User, new_user)
            case Err(msg):
                raise UserCreateError(msg)
            case _:
                raise StashError

    @as_result(StashError, NotFoundError)
    def _admin_user(self) -> User:
        # TODO: This returns only one user, the first user with the role ADMIN
        admin_credentials = self._admin_verify_key()
        return self._get_by_role(
            credentials=admin_credentials, role=ServiceRole.ADMIN
        ).unwrap()

    @as_result(StashError, NotFoundError)
    def _get_all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool = False,
    ) -> list[User]:
        result = self.partition.all(credentials, order_by, has_permission)

        match result:
            case Ok(users):
                return cast(list[User], users)
            case Err(err):
                raise NotFoundError(str(err))
            case _:
                raise StashError

    @as_result(StashError, NotFoundError)
    def _get_by_uid(self, credentials: SyftVerifyKey, uid: UID) -> User:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])

        try:
            user = self._query_one(credentials=credentials, qks=qks).unwrap()
        except NotFoundError as exc:
            msg = f"User {uid} not found"
            raise NotFoundError.from_exception(exc, public_message=msg)

        return user

    @as_result(StashError, UserDeleteError)
    def _delete_by_uid(
        self, credentials: SyftVerifyKey, uid: UID, has_permission: bool = False
    ) -> bool:
        qk = UIDPartitionKey.with_obj(uid)
        result = super().delete(
            credentials=credentials, qk=qk, has_permission=has_permission
        )

        match result:
            case Ok(_):
                return True
            case Err(err):
                msg = f"Failed to delete User {uid}"
                raise UserDeleteError(str(err), public_message=msg)
            case _:
                raise StashError

    @as_result(StashError, NotFoundError)
    def _get_by_email(self, credentials: SyftVerifyKey, email: str) -> User:
        qks = QueryKeys(qks=[EmailPartitionKey.with_obj(email)])

        try:
            user = self._query_one(credentials=credentials, qks=qks).unwrap()
        except NotFoundError as exc:
            msg = f"User {email} not found"
            raise NotFoundError.from_exception(exc, public_message=msg)

        return user

    @as_result(StashError)
    def _email_exists(self, email: str) -> bool:
        # TODO: Delete commment below, only for remembering a remark to discuss
        # In this function, stash
        try:
            self._get_by_email(
                credentials=self._admin_verify_key(), email=email
            ).unwrap()
        except NotFoundError:
            return False

        return True

    @as_result(StashError, NotFoundError)
    def _get_by_role(self, credentials: SyftVerifyKey, role: ServiceRole) -> User:
        # TODO: Is this method correct? Should'nt it return a list of all member with a particular role?
        qks = QueryKeys(qks=[RolePartitionKey.with_obj(role)])

        try:
            user = self._query_one(credentials=credentials, qks=qks).unwrap()
        except NotFoundError as exc:
            private_msg = f"User with role {role} not found"
            raise NotFoundError.from_exception(exc, private_message=private_msg)

        return user

    @as_result(StashError, NotFoundError)
    def _get_by_signing_key(
        self, credentials: SyftVerifyKey, signing_key: SyftSigningKey | str
    ) -> User:
        if isinstance(signing_key, str):
            signing_key = SyftSigningKey.from_string(signing_key)

        qks = QueryKeys(qks=[SigningKeyPartitionKey.with_obj(signing_key)])
        try:
            user = self._query_one(credentials=credentials, qks=qks).unwrap()
        except NotFoundError as exc:
            private_msg = f"User with signing key {signing_key} not found"
            raise NotFoundError.from_exception(exc, private_message=private_msg)

        return user

    @as_result(StashError, NotFoundError)
    def _get_by_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey | str
    ) -> User:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)

        qks = QueryKeys(qks=[VerifyKeyPartitionKey.with_obj(verify_key)])

        try:
            user = self._query_one(credentials=credentials, qks=qks).unwrap()
        except NotFoundError as exc:
            private_msg = f"User with verify key {verify_key} not found"
            raise NotFoundError.from_exception(exc, private_message=private_msg)

        return user

    @as_result(StashError, UserUpdateError)
    def _update(
        self, credentials: SyftVerifyKey, user: User, has_permission: bool = False
    ) -> User:
        _user = self._check_type(user, self.object_type).unwrap()

        result = super().update(
            credentials=credentials, obj=_user, has_permission=has_permission
        )

        match result:
            case Ok(update_user):
                return cast(User, update_user)
            case Err(msg):
                raise UserUpdateError(msg)
            case _:
                raise StashError

    @as_result(StashError)
    def _find_all(
        self, credentials: SyftVerifyKey, **kwargs: dict[str, Any]
    ) -> list[User]:
        result = self.query_all_kwargs(credentials=credentials, **kwargs)

        match result:
            case Ok(users):
                return cast(list[User], users)
            case Err(msg):
                raise StashError(msg)
            case _:
                raise StashError
