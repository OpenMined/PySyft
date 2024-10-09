# stdlib
from datetime import datetime
from datetime import timedelta
import secrets
import string
from typing import TypeVar
from typing import cast

# relative
from ...abstract_server import ServerType
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...store.db.db import DBManager
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...store.linked_obj import LinkedObject
from ...types.errors import CredentialsError
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_metaclass import Empty
from ...types.uid import UID
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..context import AuthedServiceContext
from ..context import ServerServiceContext
from ..context import UnauthedServiceContext
from ..notification.email_templates import OnBoardEmailTemplate
from ..notification.email_templates import PasswordResetTemplate
from ..notification.notification_service import CreateNotification
from ..notifier.notifier_enums import NOTIFIERS
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..settings.settings import PwdTokenResetConfig
from ..settings.settings_stash import SettingsStash
from .errors import UserEnclaveAdminLoginError
from .errors import UserError
from .errors import UserPermissionError
from .errors import UserUpdateError
from .user import User
from .user import UserCreate
from .user import UserPrivateKey
from .user import UserSearch
from .user import UserUpdate
from .user import UserView
from .user import check_pwd
from .user import salt_and_hash_password
from .user import validate_password
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
            _list_objs = _list_objs[page_index]  # type: ignore
        else:
            _list_objs = _list_objs[0]  # type: ignore
        return _list_objs  # type: ignore

    return list_objs


@serializable(canonical_name="UserService", version=1)
class UserService(AbstractService):
    stash: UserStash

    def __init__(self, store: DBManager) -> None:
        self.stash = UserStash(store=store)

    @as_result(StashException)
    def _add_user(self, credentials: SyftVerifyKey, user: User) -> User:
        return self.stash.set(
            credentials=credentials,
            obj=user,
            add_permissions=[
                ActionObjectPermission(
                    uid=user.id, permission=ActionPermission.ALL_READ
                ),
            ],
        ).unwrap()

    def _check_if_email_exists(self, credentials: SyftVerifyKey, email: str) -> bool:
        try:
            self.stash.get_by_email(credentials=credentials, email=email).unwrap()
            return True
        except NotFoundException:
            return False

    @service_method(path="user.create", name="create", autosplat="user_create")
    def create(
        self, context: AuthedServiceContext, user_create: UserCreate
    ) -> UserView:
        """Create a new user"""
        user = user_create.to(User)

        user_exists = self._check_if_email_exists(
            credentials=context.credentials, email=user.email
        )

        # TODO: Ensure we don't leak out the existence of a user
        if user_exists:
            raise SyftException(public_message=f"User {user.email} already exists")

        new_user = self._add_user(context.credentials, user).unwrap()
        return new_user.to(UserView)

    def forgot_password(
        self, context: UnauthedServiceContext, email: str
    ) -> SyftSuccess:
        success_msg = (
            "If the email is valid, we sent a password "
            + "reset token to your email or a password request to the admin."
        )
        root_key = self.root_verify_key

        root_context = AuthedServiceContext(server=context.server, credentials=root_key)

        result = self.stash.get_by_email(credentials=root_key, email=email)

        # Isn't a valid email
        if result.is_err():
            return SyftSuccess(message=success_msg)
        user = result.ok()

        if user is None:
            return SyftSuccess(message=success_msg)

        user_role = self.get_role_for_credentials(user.verify_key).unwrap()
        if user_role == ServiceRole.ADMIN:
            raise SyftException(
                public_message="You can't request password reset for an Admin user."
            )

        # Email is valid
        # Notifications disabled
        # We should just sent a notification to the admin/user about password reset
        # Notifications Enabled
        # Instead of changing the password here, we would change it in email template generation.
        link = LinkedObject.with_context(user, context=root_context)
        # Notifier is active
        notifier = root_context.server.services.notifier.settings(
            context=root_context
        ).unwrap()
        notification_is_enabled = notifier.active
        # Email is enabled
        email_is_enabled = notifier.email_enabled
        # User Preferences allow email notification
        user_allow_email_notifications = user.notifications_enabled[NOTIFIERS.EMAIL]

        # This checks if the user will safely receive the email reset.
        not_receive_emails = (
            not notification_is_enabled
            or not email_is_enabled
            or not user_allow_email_notifications
        )

        # If notifier service is not enabled.
        if not_receive_emails:
            message = CreateNotification(
                subject="You requested password reset.",
                from_user_verify_key=root_key,
                to_user_verify_key=user.verify_key,
                linked_obj=link,
            )
            result = root_context.server.services.notification.send(
                context=root_context, notification=message
            )
            message = CreateNotification(
                subject="User requested password reset.",
                from_user_verify_key=user.verify_key,
                to_user_verify_key=root_key,
                linked_obj=link,
            )

            result = root_context.server.services.notification.send(
                context=root_context, notification=message
            )
        else:
            # Email notification is Enabled
            # Therefore, we can directly send a message to the
            # user with its new password.
            message = CreateNotification(
                subject="You requested a password reset.",
                from_user_verify_key=root_key,
                to_user_verify_key=user.verify_key,
                linked_obj=link,
                notifier_types=[NOTIFIERS.EMAIL],
                email_template=PasswordResetTemplate,
            )
            result = root_context.server.services.notification.send(
                context=root_context, notification=message
            )

        return SyftSuccess(message=success_msg)

    @service_method(
        path="user.request_password_reset",
        name="request_password_reset",
        roles=ADMIN_ROLE_LEVEL,
    )
    def request_password_reset(self, context: AuthedServiceContext, uid: UID) -> str:
        user = self.stash.get_by_uid(credentials=context.credentials, uid=uid).unwrap()
        user_role = self.get_role_for_credentials(user.verify_key).unwrap()

        if user_role == ServiceRole.ADMIN:
            raise SyftException(
                public_message="You can't request password reset for an Admin user."
            )

        user.reset_token = self.generate_new_password_reset_token(
            context.server.settings.pwd_token_config
        )
        user.reset_token_date = datetime.now()

        self.stash.update(
            credentials=context.credentials, obj=user, has_permission=True
        ).unwrap()

        return user.reset_token

    def reset_password(
        self, context: UnauthedServiceContext, token: str, new_password: str
    ) -> SyftSuccess:
        """Resets a certain user password using a temporary token."""
        root_key = self.root_verify_key

        root_context = AuthedServiceContext(server=context.server, credentials=root_key)
        try:
            user = self.stash.get_by_reset_token(
                credentials=root_context.credentials, token=token
            ).unwrap()
        except NotFoundException:
            raise SyftException(
                public_message="Failed to reset user password. Token is invalid or expired."
            )
        #
        if user is None:
            raise SyftException(
                public_message="Failed to reset user password. Token is invalid or expired."
            )
        now = datetime.now()
        if user.reset_token_date is not None:
            time_difference = now - user.reset_token_date
        else:
            raise SyftException(
                public_message="Failed to reset user password. Reset Token Invalid!"
            )

        # If token expired
        expiration_time = root_context.server.settings.pwd_token_config.token_exp_min
        if time_difference > timedelta(seconds=expiration_time):
            raise SyftException(
                public_message="Failed to reset user password. Token is invalid or expired."
            )

        if not validate_password(new_password):
            raise SyftException(
                public_message=(
                    "Your new password must have at least 8 characters, an upper case "
                    "and lower case character; and at least one number."
                )
            )

        salt, hashed = salt_and_hash_password(new_password, 12)
        user.hashed_password = hashed
        user.salt = salt

        user.reset_token = None
        user.reset_token_date = None

        self.stash.update(
            credentials=root_context.credentials, obj=user, has_permission=True
        ).unwrap()

        return SyftSuccess(message="User Password updated successfully.")

    def generate_new_password_reset_token(
        self, token_config: PwdTokenResetConfig
    ) -> str:
        valid_characters = ""
        if token_config.ascii:
            valid_characters += string.ascii_letters

        if token_config.numbers:
            valid_characters += string.digits

        generated_token = "".join(
            secrets.choice(valid_characters) for _ in range(token_config.token_len)
        )

        return generated_token

    @service_method(path="user.view", name="view", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def view(self, context: AuthedServiceContext, uid: UID) -> UserView:
        """Get user for given uid"""
        user = self.stash.get_by_uid(credentials=context.credentials, uid=uid).unwrap()
        return user.to(UserView)

    @service_method(path="user.get_all", name="get_all", roles=DATA_OWNER_ROLE_LEVEL)
    def get_all(
        self,
        context: AuthedServiceContext,
        order_by: str | None = None,
        sort_order: str | None = None,
        page_size: int | None = 0,
        page_index: int | None = 0,
    ) -> list[UserView]:
        users = self.stash.get_all(
            context.credentials,
            order_by=order_by,
            sort_order=sort_order,
        ).unwrap()
        users = [user.to(UserView) for user in users]
        return _paginate(users, page_size, page_index)

    @service_method(
        path="user.get_index", name="get_index", roles=DATA_OWNER_ROLE_LEVEL
    )
    def get_index(
        self,
        context: AuthedServiceContext,
        index: int,
    ) -> UserView:
        return (
            self.stash.get_index(credentials=context.credentials, index=index)
            .unwrap()
            .to(UserView)
        )

    def signing_key_for_verify_key(self, verify_key: SyftVerifyKey) -> UserPrivateKey:
        user = self.stash.get_by_verify_key(
            credentials=self.stash.root_verify_key, verify_key=verify_key
        ).unwrap()

        return user.to(UserPrivateKey)

    @as_result(SyftException)
    def get_role_for_credentials(
        self, credentials: SyftVerifyKey | SyftSigningKey
    ) -> ServiceRole:
        try:
            # they could be different
            # TODO: This fn is cryptic -- when does each situation occur?
            if isinstance(credentials, SyftVerifyKey):
                role = self.stash.get_role(credentials=credentials)
                return role
            elif isinstance(credentials, SyftSigningKey):
                user = self.stash.get_by_signing_key(
                    credentials=credentials.verify_key,
                    signing_key=credentials,  # type: ignore
                ).unwrap()
            else:
                raise CredentialsError
        except NotFoundException:
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
        kwargs.pop("created_date")
        kwargs.pop("updated_date")
        kwargs.pop("deleted_date")
        if len(kwargs) == 0:
            raise SyftException(public_message="Invalid search parameters")

        users = self.stash.get_all(
            credentials=context.credentials, filters=kwargs
        ).unwrap()

        users = [user.to(UserView) for user in users] if users is not None else []
        return _paginate(users, page_size, page_index)

    @as_result(StashException, NotFoundException)
    def get_user_id_for_credentials(self, credentials: SyftVerifyKey) -> UID:
        user = self.stash.get_by_verify_key(
            credentials=credentials, verify_key=credentials
        ).unwrap()
        return cast(UID, user.id)

    @service_method(
        path="user.get_current_user", name="get_current_user", roles=GUEST_ROLE_LEVEL
    )
    def get_current_user(self, context: AuthedServiceContext) -> UserView:
        user = self.stash.get_by_verify_key(
            credentials=context.credentials, verify_key=context.credentials
        ).unwrap()
        return user.to(UserView)

    @service_method(
        path="user.get_by_verify_key", name="get_by_verify_key", roles=ADMIN_ROLE_LEVEL
    )
    def get_by_verify_key_endpoint(
        self, context: AuthedServiceContext, verify_key: SyftVerifyKey
    ) -> UserView:
        user = self.stash.get_by_verify_key(
            credentials=context.credentials, verify_key=verify_key
        ).unwrap()
        return user.to(UserView)

    @service_method(
        path="user.update",
        name="update",
        roles=GUEST_ROLE_LEVEL,
        autosplat="user_update",
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
        user = self.stash.get_by_uid(credentials=context.credentials, uid=uid).unwrap()

        immutable_fields = {"created_date", "updated_date", "deleted_date"}
        updated_fields = user_update.to_dict(
            exclude_none=True, exclude_empty=True
        ).keys()

        for field_name in immutable_fields:
            if field_name in updated_fields:
                raise SyftException(
                    public_message=f"You are not allowed to modify '{field_name}'."
                )

        # important to prevent root admins from shooting themselves in the foot
        if (
            user_update.role is not Empty  # type: ignore
            and user.verify_key == context.server.verify_key
        ):
            raise SyftException(public_message="Cannot update root role")

        if (
            user_update.verify_key is not Empty
            and user.verify_key == context.server.verify_key
        ):
            raise SyftException(public_message="Cannot update root verify key")

        if user_update.name is not Empty and user_update.name.strip() == "":  # type: ignore[comparison-overlap]
            raise SyftException(public_message="Name can't be an empty string.")

        # check if the email already exists (with root's key)
        if user_update.email is not Empty:
            user_exists = self.stash.email_exists(email=user_update.email).unwrap()
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
            if attr not in ["role", "created_date", "updated_date", "deleted_date"]
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

        user = self.stash.update(
            credentials=context.credentials, obj=user, has_permission=True
        ).unwrap()

        if user.role == ServiceRole.ADMIN:
            settings_stash = SettingsStash(store=self.stash.db)
            settings = settings_stash.get_all(
                context.credentials, limit=1, sort_order="desc"
            ).unwrap()

            # TODO: Chance to refactor here in settings, as we're always doing get_att[0]
            if len(settings) > 0:
                settings_data = settings[0]
                settings_data.admin_email = user.email
                settings_stash.update(
                    credentials=context.credentials, obj=settings_data
                )

        return user.to(UserView)

    @service_method(path="user.delete", name="delete", roles=GUEST_ROLE_LEVEL)
    def delete(self, context: AuthedServiceContext, uid: UID) -> UID:
        user_to_delete = self.stash.get_by_uid(
            credentials=context.credentials, uid=uid
        ).unwrap()

        # Cannot delete root user
        if user_to_delete.verify_key == self.root_verify_key:
            raise UserPermissionError(
                private_message=f"User {context.credentials} attempted to delete root user."
            )

        # - Admins can delete any user
        # - Data Owners can delete Data Scientists and Guests
        has_delete_permissions = (
            context.role == ServiceRole.ADMIN
            or context.role == ServiceRole.DATA_OWNER
            and user_to_delete.role in [ServiceRole.GUEST, ServiceRole.DATA_SCIENTIST]
        )

        if not has_delete_permissions:
            raise UserPermissionError(
                private_message=(
                    f"User {context.credentials} ({context.role}) tried to delete user "
                    f"{uid} ({user_to_delete.role})"
                )
            )

        # TODO: Remove notifications for the deleted user
        return self.stash.delete_by_uid(
            credentials=context.credentials, uid=uid
        ).unwrap()

    def exchange_credentials(self, context: UnauthedServiceContext) -> SyftSuccess:
        """Verify user
        TODO: We might want to use a SyftObject instead
        """

        if context.login_credentials is None:
            raise SyftException(public_message="Invalid login credentials")

        user = self.stash.get_by_email(
            credentials=self.root_verify_key, email=context.login_credentials.email
        ).unwrap()

        if check_pwd(context.login_credentials.password, user.hashed_password):
            if (
                context.server
                and context.server.server_type == ServerType.ENCLAVE
                and user.role == ServiceRole.ADMIN
            ):
                # FIX: Replace with SyftException
                raise SyftException(
                    public_message=UserEnclaveAdminLoginError.public_message
                )
        else:
            # FIX: Replace this below
            raise SyftException(public_message=CredentialsError.public_message)

        return SyftSuccess(message="Login successful.", value=user.to(UserPrivateKey))

    @property
    def root_verify_key(self) -> SyftVerifyKey:
        return self.stash.root_verify_key

    def register(
        self, context: ServerServiceContext, new_user: UserCreate
    ) -> SyftSuccess:
        """Register new user"""

        # this method handles errors in a slightly different way as it is directly called instead of
        # going through Server.handle_message

        request_user_role = (
            ServiceRole.GUEST
            if new_user.created_by is None
            else self.get_role_for_credentials(new_user.created_by).unwrap()
        )

        can_user_register = (
            context.server.settings.signup_enabled
            or request_user_role in DATA_OWNER_ROLE_LEVEL
        )

        if not can_user_register:
            raise SyftException(
                public_message="You have no permission to create an account. Please contact the Datasite owner."
            )

        user = new_user.to(User)

        user_exists = self._check_if_email_exists(
            credentials=user.verify_key, email=user.email
        )

        if user_exists:
            raise SyftException(public_message=f"User {user.email} already exists")

        user = self._add_user(credentials=user.verify_key, user=user).unwrap(
            public_message=f"Failed to create user {user.email}"
        )
        success_message = f"User '{user.name}' successfully registered!"

        # Notification Step
        root_key = self.root_verify_key
        root_context = AuthedServiceContext(server=context.server, credentials=root_key)
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
        context.server.services.notification.send(
            context=root_context, notification=message
        )

        if request_user_role in DATA_OWNER_ROLE_LEVEL:
            success_message += " To see users, run `[your_client].users`"

        return SyftSuccess(message=success_message, value=user.to(UserPrivateKey))

    @as_result(StashException)
    def user_verify_key(self, email: str) -> SyftVerifyKey:
        # we are bypassing permissions here, so dont use to return a result directly to the user
        credentials = self.root_verify_key
        user = self.stash.get_by_email(credentials=credentials, email=email).unwrap()
        if user.verify_key is None:
            raise UserError(f"User {email} has no verify key")
        return user.verify_key

    @as_result(StashException)
    def get_by_verify_key(self, verify_key: SyftVerifyKey) -> UserView:
        # we are bypassing permissions here, so dont use to return a result directly to the user
        credentials = self.root_verify_key
        user = self.stash.get_by_verify_key(
            credentials=credentials, verify_key=verify_key
        ).unwrap()
        return user.to(UserView)

    @as_result(StashException)
    def _set_notification_status(
        self,
        notifier_type: NOTIFIERS,
        new_status: bool,
        verify_key: SyftVerifyKey,
    ) -> None:
        user = self.stash.get_by_verify_key(
            credentials=verify_key, verify_key=verify_key
        ).unwrap()
        user.notifications_enabled[notifier_type] = new_status
        self.stash.update(credentials=user.verify_key, obj=user).unwrap()

    @as_result(SyftException)
    def enable_notifications(
        self, context: AuthedServiceContext, notifier_type: NOTIFIERS
    ) -> SyftSuccess:
        self._set_notification_status(
            notifier_type=notifier_type, new_status=True, verify_key=context.credentials
        ).unwrap()
        return SyftSuccess(message="Notifications enabled successfully!")

    def disable_notifications(
        self, context: AuthedServiceContext, notifier_type: NOTIFIERS
    ) -> SyftSuccess:
        self._set_notification_status(
            notifier_type=notifier_type,
            new_status=False,
            verify_key=context.credentials,
        ).unwrap()

        return SyftSuccess(message="Notifications disabled successfully!")


TYPE_TO_SERVICE[User] = UserService
SERVICE_TO_TYPES[UserService].update({User})
