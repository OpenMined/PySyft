# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from result import Err
from result import Ok

# relative
from ....core.node.common.node_table.syft_object import SyftObject
from ....telemetry import instrument
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .context import AuthedServiceContext
from .context import UnauthedServiceContext
from .credentials import UserLoginCredentials
from .document_store import DocumentStore
from .response import SyftError
from .service import AbstractService
from .service import service_method
from .user import User
from .user import UserCreate
from .user import UserPrivateKey
from .user import UserUpdate
from .user import UserView
from .user import check_pwd
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

    @service_method(path="user.find_all", name="find_all")
    def find_all(
        self, context: AuthedServiceContext, **kwargs: Dict[str, Any]
    ) -> Union[List[UserView], SyftError]:
        result = self.stash.find_all(**kwargs)
        if result.is_err():
            return SyftError(message=str(result.err()))
        users = result.ok()
        return [user.to(UserView) for user in users] if users is not None else []

    @service_method(path="user.update", name="update")
    def update(
        self, context: AuthedServiceContext, user_update: UserUpdate
    ) -> Union[UserView, SyftError]:
        user = user_update.to(User)
        result = self.stash.update(user=user)

        if result.err():
            return SyftError(message=str(result.err()))

        user = result.ok()
        return user.to(UserView)

    @service_method(path="user.delete", name="delete")
    def delete(self, context: AuthedServiceContext, uid: UID) -> Union[bool, SyftError]:

        result = self.stash.delete_by_uid(uid=uid)

        if result.err():
            return SyftError(message=str(result.err()))

        return result.ok()

    def exchange_credentials(
        self, context: UnauthedServiceContext
    ) -> Union[UserLoginCredentials, SyftError]:
        """Verify user
        TODO: We might want to use a SyftObject instead
        """
        # for _, user in self.data.items():
        # syft_object: User = SyftObject.from_mongo(user)
        # 🟡 TOD2230Store real root user and fetch from collection

        result = self.stash.get_by_email(email=context.login_credentials.email)
        if result.is_ok():
            user = result.ok()
            if user is not None and check_pwd(
                context.login_credentials.password,
                user.hashed_password,
            ):
                return Ok(user.to(UserPrivateKey))

            return Err(
                f"No user exists with {context.login_credentials.email} and supplied password."
            )

        return Err(
            f"Failed to retrieve user with {context.login_credentials.email} with error: {result.err()}"
        )

    def signup(
        self, context: UnauthedServiceContext, user_update: UserUpdate
    ) -> Union[SyftObject, SyftError]:
        pass

    # @service_method(path="user.search", name="search", splat_kwargs_from=["query_obj"])
    # def search(self, context: AuthedServiceContext, query_obj: UserQuery, limit: int):
    #     pass
