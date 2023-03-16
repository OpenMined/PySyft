# stdlib
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# relative
from ....telemetry import instrument
from .context import AuthedServiceContext
from .context import NodeServiceContext
from .context import UnauthedServiceContext
from .credentials import SyftVerifyKey
from .credentials import UserLoginCredentials
from .document_store import DocumentStore
from .response import SyftError
from .response import SyftSuccess
from .serializable import serializable
from .service import AbstractService
from .service import SERVICE_TO_TYPES
from .service import TYPE_TO_SERVICE
from .service import service_method
from .uid import UID
from .user import ServiceRole
from .user import User
from .user import UserCreate
from .user import UserPrivateKey
from .user import UserSearch
from .user import UserUpdate
from .user import UserView
from .user import check_pwd
from .user import salt_and_hash_password
from .user_stash import UserStash


@instrument
@serializable(recursive_serde=True)
class UserService(AbstractService):
    store: DocumentStore
    stash: UserStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = UserStash(store=store)

    @service_method(path="user.create", name="create")
    def create(
        self, context: AuthedServiceContext, user_create: UserCreate
    ) -> Union[UserView, SyftError]:
        """Create a new user"""
        user = user_create.to(User)
        result = self.stash.get_by_email(email=user.email)
        if result.is_err():
            return SyftError(message=str(result.err()))
        user_exists = result.ok() is not None
        if user_exists:
            return SyftError(message=f"User already exists with email: {user.email}")

        result = self.stash.set(user=user)
        if result.is_err():
            return SyftError(message=str(result.err()))
        user = result.ok()
        return user.to(UserView)

    @service_method(path="user.view", name="view")
    def view(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[Optional[UserView], SyftError]:
        """Get user for given uid"""
        result = self.stash.get_by_uid(uid=uid)
        if result.is_ok():
            user = result.ok()
            if user is None:
                return SyftError(message=f"No user exists for given: {uid}")
            return user.to(UserView)

        return SyftError(message=str(result.err()))

    @service_method(path="user.get_all", name="get_all")
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[Optional[UserView], SyftError]:
        result = self.stash.get_all()
        if result.is_ok():
            return result.ok()

        # 🟡 TODO: No user exists will happen when result.ok() is empty list
        return SyftError(message="No users exists")

    @service_method(path="user.search", name="search", autosplat=["user_search"])
    def search(
        self,
        context: AuthedServiceContext,
        user_search: UserSearch,
    ) -> Union[List[UserView], SyftError]:
        kwargs = user_search.to_dict(exclude_none=True)

        if len(kwargs) == 0:
            valid_search_params = list(UserSearch.__fields__.keys())
            return SyftError(
                message=f"Invalid Search parameters. \
                Allowed params: {valid_search_params}"
            )
        result = self.stash.find_all(**kwargs)
        if result.is_err():
            return SyftError(message=str(result.err()))
        users = result.ok()
        return [user.to(UserView) for user in users] if users is not None else []

    @service_method(path="user.update", name="update")
    def update(
        self, context: AuthedServiceContext, uid: UID, user_update: UserUpdate
    ) -> Union[UserView, SyftError]:
        # TODO: ADD Email Validation

        # Get user to be updated by its UID
        result = self.stash.get_by_uid(uid=uid)

        if result.is_err():
            error_msg = (
                f"Failed to find user with UID: {uid}. Error: {str(result.err())}"
            )
            return SyftError(message=error_msg)

        user = result.ok()

        if user is None:
            return SyftError(message=f"No user exists for given UID: {uid}")

        # Fill User Update fields that will not be changed by replacing it
        # for the current values found in user obj.
        for name, value in vars(user_update).items():
            if name == "password" and value:
                salt, hashed = salt_and_hash_password(value, 12)
                user.hashed_password = hashed
                user.salt = salt
            elif not name.startswith("__") and value is not None:
                setattr(user, name, value)

        result = self.stash.update(user=user)

        if result.is_err():
            error_msg = (
                f"Failed to update user with UID: {uid}. Error: {str(result.err())}"
            )
            return SyftError(message=error_msg)

        user = result.ok()

        return user.to(UserView)

    @service_method(path="user.delete", name="delete")
    def delete(self, context: AuthedServiceContext, uid: UID) -> Union[bool, SyftError]:
        result = self.stash.delete_by_uid(uid=uid)
        if result.is_err():
            return SyftError(message=str(result.err()))

        return result.ok()

    def exchange_credentials(
        self, context: UnauthedServiceContext
    ) -> Union[UserLoginCredentials, SyftError]:
        """Verify user
        TODO: We might want to use a SyftObject instead
        """
        result = self.stash.get_by_email(email=context.login_credentials.email)
        if result.is_ok():
            user = result.ok()
            if user is not None and check_pwd(
                context.login_credentials.password,
                user.hashed_password,
            ):
                return user.to(UserPrivateKey)

            return SyftError(
                message="No user exists with "
                f"{context.login_credentials.email} and supplied password."
            )

        return SyftError(
            message="Failed to retrieve user with "
            f"{context.login_credentials.email} with error: {result.err()}"
        )

    def admin_verify_key(self) -> Union[SyftVerifyKey, SyftError]:
        result = self.stash.get_by_role(role=ServiceRole.ADMIN)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok().verify_key

    def register(
        self, context: NodeServiceContext, new_user: UserCreate
    ) -> Union[Tuple[SyftSuccess, UserPrivateKey], SyftError]:
        """Register new user"""

        user = new_user.to(User)
        result = self.stash.get_by_email(email=user.email)
        if result.is_err():
            return SyftError(message=str(result.err()))
        user_exists = result.ok() is not None
        if user_exists:
            return SyftError(message=f"User already exists with email: {user.email}")

        result = self.stash.set(user=user)
        if result.is_err():
            return SyftError(message=str(result.err()))

        user = result.ok()
        msg = SyftSuccess(message=f"{user.email} User successfully registered !!!")
        return tuple([msg, user.to(UserPrivateKey)])

    def user_verify_key(self, email: str) -> Union[SyftVerifyKey, SyftError]:
        result = self.stash.get_by_email(email=email)
        if result.is_ok():
            return result.ok().verify_key
        return SyftError(message=f"No user with email: {email}")


TYPE_TO_SERVICE[User] = UserService
SERVICE_TO_TYPES[UserService].update({User})
