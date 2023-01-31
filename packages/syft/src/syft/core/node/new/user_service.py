# third party
from result import Err
from result import Ok
from result import Result

# relative
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.uid import UID
from .context import AuthedServiceContext
from .context import UnauthedServiceContext
from .credentials import UserLoginCredentials
from .document_store import DocumentStore
from .service import AbstractService
from .service import service_method
from .user import User
from .user import UserPrivateKey
from .user import UserUpdate
from .user import check_pwd


class UserService(AbstractService):
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.primary_keys = {}

    @service_method(path="user.create", name="create")
    def create(
        self, context: AuthedServiceContext, user_update: UserUpdate
    ) -> Result[UserUpdate, str]:
        """TEST MY DOCS"""
        if user_update.id is None:
            user_update.id = UID()
        user = user_update.to(User)

        result = self.set(context=context, uid=user.id, syft_object=user)
        if result.is_ok():
            return Ok(user.to(UserUpdate))
        else:
            return Err("Failed to create User.")

    @service_method(path="user.view", name="view")
    def view(self, context: AuthedServiceContext, uid: UID) -> Result[UserUpdate, str]:
        user_result = self.get(context=context, uid=uid)
        if user_result.is_ok():
            return Ok(user_result.ok().to(UserUpdate))
        else:
            return Err(f"Failed to get User for UID: {uid}")

    def set(
        self, context: AuthedServiceContext, uid: UID, syft_object: SyftObject
    ) -> Result[bool, str]:
        self.data[uid] = syft_object.to_mongo()
        return Ok(True)

    def exchange_credentials(
        self, context: UnauthedServiceContext
    ) -> Result[UserLoginCredentials, str]:
        """Verify user
        TODO: We might want to use a SyftObject instead
        """
        # for _, user in self.data.items():
        # syft_object: User = SyftObject.from_mongo(user)
        # 🟡 TOD 234: Store real root user and fetch from collectionO🟡
        syft_object = context.node.root_user
        if (syft_object.email == context.login_credentials.email) and check_pwd(
            context.login_credentials.password,
            syft_object.hashed_password,
        ):
            return Ok(syft_object.to(UserPrivateKey))

        return Err(
            f"No user exists with {context.login_credentials.email} and supplied password."
        )

    def get(self, context: AuthedServiceContext, uid: UID) -> Result[SyftObject, str]:
        print("self.data", self.data.keys())
        if uid not in self.data:
            return Err(f"UID: {uid} not in {type(self)} store.")
        syft_object = SyftObject.from_mongo(self.data[uid])
        return Ok(syft_object)

    def signup(
        self, context: UnauthedServiceContext, user_update: UserUpdate
    ) -> Result[SyftObject, str]:
        pass
