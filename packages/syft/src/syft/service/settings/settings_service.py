# type: ignore

# third party
from result import Err
from result import Ok

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...util.experimental_flags import flags
from ..context import AuthedServiceContext
from ..context import UnauthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..warnings import HighSideCRUDWarning
from .settings import NodeSettings
from .settings import NodeSettingsUpdate
from .settings_stash import SettingsStash


@serializable()
class SettingsService(AbstractService):
    store: DocumentStore
    stash: SettingsStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = SettingsStash(store=store)

    @service_method(path="settings.get", name="get")
    def get(self, context: UnauthedServiceContext) -> NodeSettings | SyftError:
        """
        Get the Node Settings
        Returns:
            NodeSettings | SyftError : The Node Settings or an error if no settings are found.
        """

        result = self.stash.get(context.node.signing_key.verify_key)

        match result:  # type: ignore
            case Ok(None):
                return SyftError(message="No settings found")
            case Ok(NodeSettings() as settings):
                return settings
            case Err(err_message):
                return SyftError(message=err_message)

    @service_method(path="settings.set", name="set")
    def set(
        self, context: AuthedServiceContext, settings: NodeSettings
    ) -> NodeSettings | SyftError:
        """
        Set a new the Node Settings
        Returns:
            NodeSettings | SyftError : The new Node Settings or an error if the settings could not be set.
        """
        result = self.stash.set(context.credentials, settings)

        match result:
            case Ok(settings):
                return settings
            case Err(err_message):
                return SyftError(message=err_message)

    @service_method(path="settings.update", name="update", autosplat=["settings"])
    def update(
        self, context: AuthedServiceContext, settings: NodeSettingsUpdate
    ) -> SyftSuccess | SyftError:
        """
        Update the Node Settings using the provided values.

        Args:
            name: Optional[str]
                Node name
            organization: Optional[str]
                Organization name
            description: Optional[str]
                Node description
            on_board: Optional[bool]
                Show onboarding panel when a user logs in for the first time
            signup_enabled: Optional[bool]
                Enable/Disable registration
            admin_email: Optional[str]
                Administrator email
            association_request_auto_approval: Optional[bool]

        Returns:
            SyftSuccess | SyftError: A result indicating the success or failure of the update operation.

        Example:
        >>> node_client.update(name='foo', organization='bar', description='baz', signup_enabled=True)
        SyftSuccess: Settings updated successfully.
        """

        result = self.get(context)
        match result:  # type: ignore
            case NodeSettings():
                new_settings = result.model_copy(
                    update=settings.to_dict(exclude_empty=True)
                )
                update_result = self.stash.update(context.credentials, new_settings)
                match update_result:
                    case Ok():
                        return SyftSuccess(
                            message=(
                                "Settings updated successfully. "
                                + "You must call <client>.refresh() to sync your client with the changes."
                            )
                        )
                    case Err(err_message):
                        return SyftError(message=err_message)
            case SyftError():
                return result

    @service_method(
        path="settings.enable_notifications",
        name="enable_notifications",
        roles=ADMIN_ROLE_LEVEL,
    )
    def enable_notifications(
        self,
        context: AuthedServiceContext,
        email_username: str | None = None,
        email_password: str | None = None,
        email_sender: str | None = None,
        email_server: str | None = None,
        email_port: str | None = None,
    ) -> SyftSuccess | SyftError:
        notifier_service = context.node.get_service("notifierservice")
        return notifier_service.turn_on(
            context=context,
            email_username=email_username,
            email_password=email_password,
            email_sender=email_sender,
            email_server=email_server,
            email_port=email_port,
        )

    @service_method(
        path="settings.disable_notifications",
        name="disable_notifications",
        roles=ADMIN_ROLE_LEVEL,
    )
    def disable_notifications(
        self,
        context: AuthedServiceContext,
    ) -> SyftSuccess | SyftError:
        notifier_service = context.node.get_service("notifierservice")
        return notifier_service.turn_off(context=context)

    @service_method(
        path="settings.allow_guest_signup",
        name="allow_guest_signup",
        warning=HighSideCRUDWarning(confirmation=True),
    )
    def allow_guest_signup(
        self, context: AuthedServiceContext, enable: bool
    ) -> SyftSuccess | SyftError:
        """Enable/Disable Registration for Data Scientist or Guest Users."""
        flags.CAN_REGISTER = enable

        new_settings = NodeSettingsUpdate(signup_enabled=enable)
        result = self.update(context, settings=new_settings)

        match result:  # type: ignore
            case SyftSuccess():
                flag = "enabled" if enable else "disabled"
                return SyftSuccess(message=f"Registration feature successfully {flag}")
            case SyftError():
                return SyftError(message=f"Failed to update settings: {result}")

    @service_method(
        path="settings.allow_association_request_auto_approval",
        name="allow_association_request_auto_approval",
    )
    def allow_association_request_auto_approval(
        self, context: AuthedServiceContext, enable: bool
    ) -> SyftSuccess | SyftError:
        new_settings = NodeSettingsUpdate(association_request_auto_approval=enable)
        result = self.update(context, settings=new_settings)

        match result:  # type: ignore
            case SyftSuccess():
                flag = "enabled" if enable else "disabled"
                return SyftSuccess(
                    message=f"Association request auto-approval successfully {flag}"
                )
            case SyftError():
                return SyftError(message=f"Failed to update settings: {result}")
