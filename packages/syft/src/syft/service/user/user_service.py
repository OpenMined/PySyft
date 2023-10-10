# stdlib
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# relative
from ...abstract_node import NodeType
from ...exceptions.user import AdminEnclaveLoginException
from ...exceptions.user import AdminVerifyKeyException
from ...exceptions.user import DeleteUserPermissionsException
from ...exceptions.user import FailedToUpdateUserWithUIDException
from ...exceptions.user import GenericException
from ...exceptions.user import InvalidSearchParamsException
from ...exceptions.user import NoUserFoundException
from ...exceptions.user import NoUserWithEmailException
from ...exceptions.user import NoUserWithUIDException
from ...exceptions.user import NoUserWithVerifyKeyException
from ...exceptions.user import RegisterUserPermissionsException
from ...exceptions.user import RoleNotAllowedToEditRolesException
from ...exceptions.user import RoleNotAllowedToEditSpecificRolesException
from ...exceptions.user import StashRetrievalException
from ...exceptions.user import UserWithEmailAlreadyExistsException
from ...node.credentials import SyftSigningKey
from ...node.credentials import SyftVerifyKey
from ...node.credentials import UserLoginCredentials
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.syft_metaclass import Empty
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..context import AuthedServiceContext
from ..context import NodeServiceContext
from ..context import UnauthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..settings.settings_stash import SettingsStash
from .user import User
from .user import UserCreate
from .user import UserPrivateKey
from .user import UserSearch
from .user import UserUpdate
from .user import UserView
from .user import UserViewPage
from .user import check_pwd
from .user import salt_and_hash_password
from .user_roles import DATA_OWNER_ROLE_LEVEL
from .user_roles import GUEST_ROLE_LEVEL
from .user_roles import ServiceRole
from .user_roles import ServiceRoleCapability
from .user_stash import UserStash


@instrument
@serializable()
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
        result = self.stash.get_by_email(
            credentials=context.credentials, email=user.email
        )
        if result.is_err():
            raise NoUserWithEmailException(
                email=user.email, err=result.err()
            ).raise_with_context(context=context)
        user_exists = result.ok() is not None
        if user_exists:
            raise UserWithEmailAlreadyExistsException(
                email=user.email
            ).raise_with_context(context=context)

        result = self.stash.set(
            credentials=context.credentials,
            user=user,
            add_permissions=[
                ActionObjectPermission(
                    uid=user.id, permission=ActionPermission.ALL_READ
                ),
            ],
        )
        if result.is_err():
            raise GenericException(message=str(result.err())).raise_with_context(
                context=context
            )
        user = result.ok()
        return user.to(UserView)

    @service_method(path="user.view", name="view")
    def view(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[Optional[UserView], SyftError]:
        """Get user for given uid"""
        result = self.stash.get_by_uid(credentials=context.credentials, uid=uid)
        if result.is_ok():
            user = result.ok()
            if user is None:
                raise NoUserWithUIDException(uid=uid).raise_with_context(
                    context=context
                )
            return user.to(UserView)

        raise GenericException(message=str(result.err())).raise_with_context(
            context=context
        )

    @service_method(
        path="user.get_all",
        name="get_all",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_all(
        self,
        context: AuthedServiceContext,
        page_size: Optional[int] = 0,
        page_index: Optional[int] = 0,
    ) -> Union[Optional[UserViewPage], Optional[UserView], SyftError]:
        if context.role in [ServiceRole.DATA_OWNER, ServiceRole.ADMIN]:
            result = self.stash.get_all(context.credentials, has_permission=True)
        else:
            result = self.stash.get_all(context.credentials)
        if result.is_ok():
            results = [user.to(UserView) for user in result.ok()]

            # If chunk size is defined, then split list into evenly sized chunks
            if page_size:
                total = len(results)
                results = [
                    results[i : i + page_size]
                    for i in range(0, len(results), page_size)
                ]
                # Return the proper slice using chunk_index
                results = results[page_index]
                results = UserViewPage(users=results, total=total)
            return results

        raise NoUserFoundException.raise_with_context(context=context)

    def get_role_for_credentials(
        self, credentials: Union[SyftVerifyKey, SyftSigningKey]
    ) -> Union[Optional[ServiceRole], SyftError]:
        # they could be different

        if isinstance(credentials, SyftVerifyKey):
            result = self.stash.get_by_verify_key(
                credentials=credentials, verify_key=credentials
            )
        else:
            result = self.stash.get_by_signing_key(
                credentials=credentials, signing_key=credentials
            )
        if result.is_ok():
            # this seems weird that we get back None as Ok(None)
            user = result.ok()
            if user:
                return user.role
        return ServiceRole.GUEST

    @service_method(path="user.search", name="search", autosplat=["user_search"])
    def search(
        self,
        context: AuthedServiceContext,
        user_search: UserSearch,
        page_size: Optional[int] = 0,
        page_index: Optional[int] = 0,
    ) -> Union[Optional[UserViewPage], List[UserView], SyftError]:
        kwargs = user_search.to_dict(exclude_empty=True)

        if len(kwargs) == 0:
            valid_search_params = list(UserSearch.__fields__.keys())
            raise InvalidSearchParamsException(
                valid_search_params=valid_search_params
            ).raise_with_context(context=context)
        result = self.stash.find_all(credentials=context.credentials, **kwargs)

        if result.is_err():
            raise GenericException(message=str(result.err())).raise_with_context(
                context=context
            )
        users = result.ok()
        results = [user.to(UserView) for user in users] if users is not None else []

        # If page size is defined, then split list into evenly sized chunks
        if page_size:
            total = len(results)
            results = [
                results[i : i + page_size] for i in range(0, len(results), page_size)
            ]
            # Return the proper slice using page_index
            results = results[page_index]
            results = UserViewPage(users=results, total=total)

        return results

    # @service_method(path="user.get_admin", name="get_admin", roles=GUEST_ROLE_LEVEL)
    # def get_admin(self, context: AuthedServiceContext) -> UserView:
    #     result = self.stash.admin_user()
    #     if result.is_ok():
    #         user = result.ok()
    #         if user:
    #             return user
    #     return SyftError(message=str(result.err()))

    @service_method(
        path="user.get_current_user", name="get_current_user", roles=GUEST_ROLE_LEVEL
    )
    def get_current_user(
        self, context: AuthedServiceContext
    ) -> Union[UserView, SyftError]:
        result = self.stash.get_by_verify_key(
            credentials=context.credentials, verify_key=context.credentials
        )
        if result.is_ok():
            # this seems weird that we get back None as Ok(None)
            user = result.ok()
            if user:
                return user.to(UserView)
            else:
                raise NoUserFoundException.raise_with_context(context=context)
        raise GenericException(message=str(result.err())).raise_with_context(
            context=context
        )

    @service_method(
        path="user.update",
        name="update",
        roles=GUEST_ROLE_LEVEL,
    )
    def update(
        self, context: AuthedServiceContext, uid: UID, user_update: UserUpdate
    ) -> Union[UserView, SyftError]:
        updates_role = user_update.role is not Empty

        if (
            updates_role
            and ServiceRoleCapability.CAN_EDIT_ROLES not in context.capabilities()
        ):
            raise RoleNotAllowedToEditRolesException(
                role=context.role
            ).raise_with_context(context=context)

        # Get user to be updated by its UID
        result = self.stash.get_by_uid(credentials=context.credentials, uid=uid)

        # check if the email already exists (with root's key)
        if user_update.email is not Empty:
            user_with_email_exists: bool = self.stash.email_exists(
                email=user_update.email
            )
            if user_with_email_exists:
                raise UserWithEmailAlreadyExistsException(
                    email=user_update.email
                ).raise_with_context(context=context)

        if result.is_err():
            raise NoUserWithUIDException(uid=uid).raise_with_context(context=context)

        user = result.ok()

        if user is None:
            raise NoUserWithUIDException(uid=uid).raise_with_context(context=context)

        if updates_role:
            if context.role == ServiceRole.ADMIN:
                # do anything
                pass
            elif (
                context.role == ServiceRole.DATA_OWNER
                and context.role > user.role
                and context.role > user_update.role
            ):
                # as a data owner, only update lower roles to < data owner
                pass
            else:
                raise RoleNotAllowedToEditSpecificRolesException(
                    ctx_role=context.role,
                    user_role=user.role,
                    user_update_role=user_update.role,
                ).raise_with_context(context=context)

        edits_non_role_attrs = any(
            getattr(user_update, attr) is not Empty
            for attr in user_update.dict()
            if attr != "role"
        )

        if (
            edits_non_role_attrs
            and user.verify_key != context.credentials
            and ServiceRoleCapability.CAN_MANAGE_USERS not in context.capabilities()
        ):
            raise RoleNotAllowedToEditSpecificRolesException(
                ctx_role=context.role, user_role="users"
            ).raise_with_context(context=context)

        # Fill User Update fields that will not be changed by replacing it
        # for the current values found in user obj.
        for name, value in user_update.to_dict(exclude_empty=True).items():
            if name == "password" and value:
                salt, hashed = salt_and_hash_password(value, 12)
                user.hashed_password = hashed
                user.salt = salt
            elif not name.startswith("__") and value is not None:
                setattr(user, name, value)

        result = self.stash.update(
            credentials=context.credentials, user=user, has_permission=True
        )

        if result.is_err():
            raise FailedToUpdateUserWithUIDException(
                uid=uid, err=str(result.err())
            ).raise_with_context(context=context)

        user = result.ok()
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

    def get_target_object(self, credentials: SyftVerifyKey, uid: UID):
        user_result = self.stash.get_by_uid(credentials=credentials, uid=uid)
        if user_result.is_err():
            return SyftError(message=str(user_result.err()))
        user = user_result.ok()
        if user is None:
            raise NoUserWithUIDException(uid=uid)
        else:
            return user

    @service_method(path="user.delete", name="delete", roles=GUEST_ROLE_LEVEL)
    def delete(self, context: AuthedServiceContext, uid: UID) -> Union[bool, SyftError]:
        # third party
        user = self.get_target_object(context.credentials, uid)
        if isinstance(user, SyftError):
            return user

        if context.role == ServiceRole.DATA_OWNER and user.role in [
            ServiceRole.GUEST,
            ServiceRole.DATA_SCIENTIST,
        ]:
            pass
        elif context.role == ServiceRole.ADMIN:
            pass
        else:
            raise DeleteUserPermissionsException(
                user_role=context.role, target_role=user.role
            ).raise_with_context(context=context)

        result = self.stash.delete_by_uid(
            credentials=context.credentials, uid=uid, has_permission=True
        )
        if result.is_err():
            raise GenericException(message=str(result.err())).raise_with_context(
                context=context
            )

        return result.ok()

    def exchange_credentials(
        self, context: UnauthedServiceContext
    ) -> Union[UserLoginCredentials, SyftError]:
        """Verify user
        TODO: We might want to use a SyftObject instead
        """
        result = self.stash.get_by_email(
            credentials=self.admin_verify_key(), email=context.login_credentials.email
        )
        if result.is_ok():
            user = result.ok()
            if user is not None and check_pwd(
                context.login_credentials.password,
                user.hashed_password,
            ):
                if (
                    context.node.node_type == NodeType.ENCLAVE
                    and user.role == ServiceRole.ADMIN
                ):
                    raise AdminEnclaveLoginException.raise_with_context(context=context)
                return user.to(UserPrivateKey)

            raise NoUserWithEmailException(
                email=context.login_credentials.email
            ).raise_with_context(context=context)

        raise NoUserWithEmailException(
            email=context.login_credentials.email, err=result.err()
        ).raise_with_context(context=context)

    def register(
        self, context: NodeServiceContext, new_user: UserCreate
    ) -> Union[Tuple[SyftSuccess, UserPrivateKey], SyftError]:
        """Register new user"""

        request_user_role = (
            ServiceRole.GUEST
            if new_user.created_by is None
            else self.get_role_for_credentials(new_user.created_by)
        )
        can_user_register = (
            context.node.metadata.signup_enabled
            or request_user_role in DATA_OWNER_ROLE_LEVEL
        )

        if not can_user_register:
            raise RegisterUserPermissionsException(
                domain=context.node.name
            ).raise_with_context(context=context)

        user = new_user.to(User)
        result = self.stash.get_by_email(credentials=user.verify_key, email=user.email)
        if result.is_err():
            return SyftError(message=str(result.err()))
        user_exists = result.ok() is not None
        if user_exists:
            raise UserWithEmailAlreadyExistsException(
                email=user.email
            ).raise_with_context(context=context)

        result = self.stash.set(
            credentials=user.verify_key,
            user=user,
            add_permissions=[
                ActionObjectPermission(
                    uid=user.id, permission=ActionPermission.ALL_READ
                ),
            ],
        )
        if result.is_err():
            raise GenericException(message=str(result.err())).raise_with_context(
                context=context
            )

        user = result.ok()

        success_message = f"User '{user.name}' successfully registered!"
        if request_user_role in DATA_OWNER_ROLE_LEVEL:
            success_message += " To see users, run `[your_client].users`"
        msg = SyftSuccess(message=success_message)
        return (msg, user.to(UserPrivateKey))

    def admin_verify_key(self) -> SyftVerifyKey:
        try:
            result = self.stash.admin_verify_key()
        except Exception as e:
            raise StashRetrievalException(message=repr(e)) from e

        if result.is_ok() and result.ok() is not None:
            return result.ok()
        else:
            raise AdminVerifyKeyException

    def user_verify_key(self, email: str) -> SyftVerifyKey:
        # we are bypassing permissions here, so dont use to return a result directly to the user
        credentials = self.admin_verify_key()
        try:
            result = self.stash.get_by_email(credentials=credentials, email=email)
        except Exception as e:
            raise StashRetrievalException(message=repr(e)) from e

        if result.is_ok() and result.ok() is not None:
            return result.ok().verify_key
        else:
            raise NoUserWithEmailException(email)

    def get_by_verify_key(self, verify_key: SyftVerifyKey) -> User:
        # we are bypassing permissions here, so dont use to return a result directly to the user
        credentials = self.admin_verify_key()
        try:
            result = self.stash.get_by_verify_key(
                credentials=credentials, verify_key=verify_key
            )
        except Exception as e:
            raise StashRetrievalException(message=repr(e)) from e

        if result.is_ok() and result.ok() is not None:
            return result.ok()
        else:
            raise NoUserWithVerifyKeyException(verify_key.verify)


TYPE_TO_SERVICE[User] = UserService
SERVICE_TO_TYPES[UserService].update({User})
