# stdlib
from ast import TypeVar
from typing import cast

# third party
from IPython.display import display_html

# relative
from ...abstract_node import NodeType
from ...node.credentials import SyftSigningKey
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.errors import NotFoundError
from ...store.errors import StashError
from ...store.linked_obj import LinkedObject
from ...types.errors import CredentialsError
from ...types.result import as_result
from ...types.syft_metaclass import Empty
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..context import AuthedServiceContext
from ..context import NodeServiceContext
from ..context import UnauthedServiceContext
from ..notification.email_templates import OnBoardEmailTemplate
from ..notification.notification_service import CreateNotification
from ..notification.notification_service import NotificationService
from ..notifier.notifier_enums import NOTIFIERS
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..settings.settings_stash import SettingsStash
from .errors import UserCreateError
from .errors import UserEnclaveAdminLoginError
from .errors import UserError
from .errors import UserPermissionError
from .errors import UserSearchBadParamsError
from .errors import UserUpdateError
from .user import User
from .user import UserCreate
from .user import UserPrivateKey
from .user import UserSearch
from .user import UserUpdate
from .user import UserView
from .user import check_pwd
from .user import salt_and_hash_password
from .user_roles import ADMIN_ROLE_LEVEL
from .user_roles import DATA_OWNER_ROLE_LEVEL
from .user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .user_roles import GUEST_ROLE_LEVEL
from .user_roles import ServiceRole
from .user_roles import ServiceRoleCapability
from .user_stash import UserStash

T = TypeVar("T")


def _paginate(
    list_objs: list[T], page_size: int | None = 0, page_index: int | None = 0
) -> list[T]:
    # If chunk size is defined, then split list into evenly sized chunks
    if page_size:
        _list_objs = [
            list_objs[i : i + page_size] for i in range(0, len(list_objs), page_size)
        ]

        # Return the proper slice using chunk_index
        if page_index is not None:
            _list_objs = _list_objs[page_index]
        else:
            _list_objs = _list_objs[0]
        return _list_objs

    return list_objs


@instrument
@serializable()
class UserService(AbstractService):
    store: DocumentStore
    stash: UserStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = UserStash(store=store)

    def _add_user(self, credentials: SyftVerifyKey, user: User) -> User:
        action_object_permissions = ActionObjectPermission(
            uid=user.id, permission=ActionPermission.ALL_READ
        )

        return self.stash._set(
            credentials=credentials,
            user=user,
            add_permissions=[action_object_permissions],
        ).unwrap()

    def _check_if_email_exists(self, credentials: SyftVerifyKey, email: str) -> bool:
        try:
            self.stash._get_by_email(credentials=credentials, email=email).unwrap()
            return True
        except NotFoundError:
            return False

    @service_method(path="user.create", name="create")
    def create(
        self, context: AuthedServiceContext, user_create: UserCreate
    ) -> UserView:
        """Create a new user"""
        user = user_create.to(User)

        user_exists = self._check_if_email_exists(
            credentials=context.credentials, email=user.email
        )

        if not user_exists:
            new_user = self._add_user(context.credentials, user)
            return new_user.to(UserView)

        raise UserCreateError(f"User {user.email} already exists")

    @service_method(path="user.view", name="view", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def view(self, context: AuthedServiceContext, uid: UID) -> UserView:
        """Get user for given uid"""
        user = self.stash._get_by_uid(credentials=context.credentials, uid=uid).unwrap()
        return user.to(UserView)

    @service_method(path="user.get_all", name="get_all", roles=DATA_OWNER_ROLE_LEVEL)
    def get_all(
        self,
        context: AuthedServiceContext,
        page_size: int | None = 0,
        page_index: int | None = 0,
    ) -> list[UserView]:
        if context.role in [ServiceRole.DATA_OWNER, ServiceRole.ADMIN]:
            users = self.stash._get_all(
                context.credentials, has_permission=True
            ).unwrap()
        else:
            users = self.stash._get_all(context.credentials).unwrap()
        users = [user.to(UserView) for user in users]
        return _paginate(users, page_size, page_index)

    def signing_key_for_verify_key(
        self, verify_key: SyftVerifyKey
    ) -> UserPrivateKey | SyftError:
        user = self.stash._get_by_verify_key(
            credentials=self.stash._admin_verify_key(), verify_key=verify_key
        ).unwrap()

        return user.to(UserPrivateKey)

    def get_role_for_credentials(
        self, credentials: SyftVerifyKey | SyftSigningKey
    ) -> ServiceRole:
        try:
            # they could be different
            # TODO: This fn is cryptic -- when does each situation occur?
            if isinstance(credentials, SyftVerifyKey):
                user = self.stash._get_by_verify_key(
                    credentials=credentials, verify_key=credentials
                ).unwrap()
            elif isinstance(credentials, SyftSigningKey):
                user = self.stash._get_by_signing_key(
                    credentials=credentials,
                    signing_key=credentials,  # type: ignore
                ).unwrap()
            else:
                raise CredentialsError
        except NotFoundError:
            return ServiceRole.GUEST

        return cast(ServiceRole, user.role)

    @service_method(path="user.search", name="search", autosplat=["user_search"])
    def search(
        self,
        context: AuthedServiceContext,
        user_search: UserSearch,
        page_size: int | None = 0,
        page_index: int | None = 0,
    ) -> list[UserView]:
        kwargs = user_search.to_dict(exclude_empty=True)

        if len(kwargs) == 0:
            raise UserSearchBadParamsError

        _users = self.stash._find_all(
            credentials=context.credentials, **kwargs
        ).unwrap()
        _users = [user.to(UserView) for user in _users] if _users is not None else []
        return _paginate(_users, page_size, page_index)

    @as_result(StashError, NotFoundError)
    def get_user_id_for_credentials(self, credentials: SyftVerifyKey) -> UID:
        user = self.stash._get_by_verify_key(
            credentials=credentials, verify_key=credentials
        ).unwrap()
        return cast(UID, user.id)

    @service_method(
        path="user.get_current_user", name="get_current_user", roles=GUEST_ROLE_LEVEL
    )
    def get_current_user(self, context: AuthedServiceContext) -> UserView:
        user = self.stash._get_by_verify_key(
            credentials=context.credentials, verify_key=context.credentials
        ).unwrap()
        return user.to(UserView)

    @service_method(
        path="user.get_by_verify_key", name="get_by_verify_key", roles=ADMIN_ROLE_LEVEL
    )
    def get_by_verify_key_endpoint(
        self, context: AuthedServiceContext, verify_key: SyftVerifyKey
    ) -> UserView:
        user = self.stash._get_by_verify_key(
            credentials=context.credentials, verify_key=verify_key
        ).unwrap()
        return user.to(UserView)

    @service_method(
        path="user.update",
        name="update",
        roles=GUEST_ROLE_LEVEL,
    )
    def update(
        self, context: AuthedServiceContext, uid: UID, user_update: UserUpdate
    ) -> UserView:
        updates_role = user_update.role is not Empty  # type: ignore[comparison-overlap]
        can_edit_roles = ServiceRoleCapability.CAN_EDIT_ROLES in context.capabilities()

        if updates_role and not can_edit_roles:
            raise UserPermissionError(
                f"User {context.credentials} tried to update user {uid} with {user_update}."
            )
        if (user_update.mock_execution_permission is not Empty) and not can_edit_roles:  # type: ignore[comparison-overlap]
            raise UserPermissionError(
                f"User {context.credentials} with role {context.role} is not allowed"
                " to update permissions."
            )

        # Get user to be updated by its UID
        user = self.stash._get_by_uid(credentials=context.credentials, uid=uid).unwrap()

        # check if the email already exists (with root's key)
        if user_update.email is not Empty:
            user_exists = self.stash._email_exists(email=user_update.email).unwrap()
            if user_exists:
                raise UserUpdateError(
                    public_message=f"User {user_update.email} already exists"
                )

        if updates_role:
            if context.role == ServiceRole.ADMIN:
                # do anything
                pass
            elif (
                context.role == ServiceRole.DATA_OWNER
                and user.role is not None
                and context.role.value > user.role.value
                and context.role.value > user_update.role.value
            ):
                # as a data owner, only update lower roles to < data owner
                pass
            else:
                raise UserPermissionError(
                    f"User {context.credentials} tried to update user {uid}"
                    f" with {user_update}."
                )

        edits_non_role_attrs = any(
            getattr(user_update, attr) is not Empty
            for attr in user_update.to_dict()
            if attr != "role"
        )

        if (
            edits_non_role_attrs
            and user.verify_key != context.credentials
            and ServiceRoleCapability.CAN_MANAGE_USERS not in context.capabilities()
        ):
            raise UserPermissionError(
                f"User {context.credentials} tried to update user {uid}"
                f" with {user_update}."
            )

        # Fill User Update fields that will not be changed by replacing it
        # for the current values found in user obj.
        for name, value in user_update.to_dict(exclude_empty=True).items():
            if name == "password" and value:
                salt, hashed = salt_and_hash_password(value, 12)
                user.hashed_password = hashed
                user.salt = salt
            elif not name.startswith("__") and value is not None:
                setattr(user, name, value)

        user = self.stash._update(
            credentials=context.credentials, user=user, has_permission=True
        ).unwrap()

        if user.role == ServiceRole.ADMIN:
            settings_stash = SettingsStash(store=self.store)
            settings = settings_stash.get_all(context.credentials)
            if settings.is_ok() and len(settings.ok()) > 0:
                settings_data = settings.ok()[0]
                settings_data.admin_email = user.email
                settings_stash.update(
                    credentials=context.credentials, settings=settings_data
                )

        return user.to(UserView)

    @service_method(path="user.delete", name="delete", roles=GUEST_ROLE_LEVEL)
    def delete(self, context: AuthedServiceContext, uid: UID) -> bool:
        user = self.stash._get_by_uid(credentials=context.credentials, uid=uid).unwrap()

        if (
            context.role == ServiceRole.ADMIN
            or context.role == ServiceRole.DATA_OWNER
            and user.role
            in [
                ServiceRole.GUEST,
                ServiceRole.DATA_SCIENTIST,
            ]
        ):
            pass
        else:
            raise UserPermissionError(
                f"User {context.credentials} ({context.role}) tried to delete user {uid} ({user.role})"
            )

        # TODO: Remove notifications for the deleted user

        return self.stash._delete_by_uid(
            credentials=context.credentials, uid=uid, has_permission=True
        ).unwrap()

    def exchange_credentials(self, context: UnauthedServiceContext) -> UserPrivateKey:
        """Verify user
        TODO: We might want to use a SyftObject instead
        """
        user = self.stash._get_by_email(
            credentials=self.admin_verify_key(), email=context.login_credentials.email
        ).unwrap()

        if check_pwd(context.login_credentials.password, user.hashed_password):
            if (
                context.node
                and context.node.node_type == NodeType.ENCLAVE
                and user.role == ServiceRole.ADMIN
            ):
                # TODO: Entertain an auth service?
                raise UserEnclaveAdminLoginError
        else:
            raise CredentialsError

        return user.to(UserPrivateKey)

    def admin_verify_key(self) -> SyftVerifyKey:
        # TODO: Remove passthrough method?
        return self.stash._admin_verify_key()

    def register(
        self, context: NodeServiceContext, new_user: UserCreate
    ) -> UserPrivateKey:
        """Register new user"""
        request_user_role = (
            ServiceRole.GUEST
            if new_user.created_by is None
            else self.get_role_for_credentials(new_user.created_by)
        )

        can_user_register = (
            context.node.settings.signup_enabled
            or request_user_role in DATA_OWNER_ROLE_LEVEL
        )

        if not can_user_register:
            raise UserPermissionError

        user = new_user.to(User)

        user_exists = self._check_if_email_exists(
            credentials=user.verify_key, email=user.email
        )

        if user_exists:
            raise UserCreateError(f"User {user.email} already exists")

        user = self._add_user(credentials=user.verify_key, user=user)

        success_message = f"User '{user.name}' successfully registered!"

        # Notification Step
        root_key = self.admin_verify_key()
        root_context = AuthedServiceContext(node=context.node, credentials=root_key)
        link = None

        if new_user.created_by:
            link = LinkedObject.with_context(user, context=root_context)

        message = CreateNotification(
            subject=success_message,
            from_user_verify_key=root_key,
            to_user_verify_key=user.verify_key,
            linked_obj=link,
            notifier_types=[NOTIFIERS.EMAIL],
            email_template=OnBoardEmailTemplate,
        )

        method = context.node.get_service_method(NotificationService.send)
        method(context=root_context, notification=message)

        if request_user_role in DATA_OWNER_ROLE_LEVEL:
            success_message += " To see users, run `[your_client].users`"

        # TODO: Add a notification for the new user
        msg = SyftSuccess(message=success_message)
        display_html(msg)

        return user.to(UserPrivateKey)

    def user_verify_key(self, email: str) -> SyftVerifyKey:
        # we are bypassing permissions here, so dont use to return a result directly to the user
        credentials = self.admin_verify_key()
        user = self.stash._get_by_email(credentials=credentials, email=email).unwrap()
        if user.verify_key is None:
            raise UserError(f"User {email} has no verify key")
        return user.verify_key

    def get_by_verify_key(self, verify_key: SyftVerifyKey) -> UserView:
        # we are bypassing permissions here, so dont use to return a result directly to the user
        credentials = self.admin_verify_key()
        user = self.stash._get_by_verify_key(
            credentials=credentials, verify_key=verify_key
        ).unwrap()
        return user.to(UserView)

    def _set_notification_status(
        self,
        notifier_type: NOTIFIERS,
        new_status: bool,
        verify_key: SyftVerifyKey,
    ) -> None:
        user = self.stash._get_by_verify_key(
            credentials=verify_key, verify_key=verify_key
        ).unwrap()
        user.notifications_enabled[notifier_type] = new_status
        self.stash._update(credentials=user.verify_key, user=user).unwrap()

    def enable_notifications(
        self, context: AuthedServiceContext, notifier_type: NOTIFIERS
    ) -> SyftSuccess:
        self._set_notification_status(
            notifier_type=notifier_type, new_status=True, verify_key=context.credentials
        )
        return SyftSuccess(message="Notifications enabled successfully!")

    def disable_notifications(
        self, context: AuthedServiceContext, notifier_type: NOTIFIERS
    ) -> SyftSuccess:
        self._set_notification_status(
            notifier_type=notifier_type,
            new_status=False,
            verify_key=context.credentials,
        )

        return SyftSuccess(message="Notifications disabled successfully!")


TYPE_TO_SERVICE[User] = UserService
SERVICE_TO_TYPES[UserService].update({User})
