# stdlib
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .context import AuthedServiceContext
from .context import UnauthedServiceContext
from .credentials import UserLoginCredentials
from .document_store import DocumentStore
from .service import AbstractService
from .service import service_method
from .user import User
from .user import UserCreate
from .user import UserPrivateKey
from .user import UserUpdate
from .user import UserView
from .user import check_pwd
from .user_stash import UserStash


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
    ) -> Result[UserView, str]:
        """Create a new user"""
        user = user_create.to(User)
        return self.stash.set(user=user).map(lambda x: x.to(UserView))

    @service_method(path="user.view", name="view")
    def view(
        self, context: AuthedServiceContext, uid: UID
    ) -> Result[Optional[UserView], str]:
        """Get user for given uid"""
        result = self.stash.get_by_uid(uid=uid)
        return result.ok().map(lambda x: x if x is None else x.to(UserView))

    def exchange_credentials(
        self, context: UnauthedServiceContext
    ) -> Result[UserLoginCredentials, str]:
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
    ) -> Result[SyftObject, str]:
        pass

    # @service_method(path="user.search", name="search", splat_kwargs_from=["query_obj"])
    # def search(self, context: AuthedServiceContext, query_obj: UserQuery, limit: int):
    #     pass
