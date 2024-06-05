# stdlib
from string import Template

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...abstract_node import NodeSideType
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...util.assets import load_png_base64
from ...util.experimental_flags import flags
from ...util.misc_objs import HTMLObject
from ...util.misc_objs import MarkdownDescription
from ...util.notebook_ui.styles import FONT_CSS
from ...util.schema import DO_COMMANDS
from ...util.schema import DS_COMMANDS
from ...util.schema import GUEST_COMMANDS
from ..context import AuthedServiceContext
from ..context import UnauthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..user.user_roles import ServiceRole
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
    def get(self, context: UnauthedServiceContext) -> Result[Ok, Err]:
        """Get Settings"""

        result = self.stash.get_all(context.node.signing_key.verify_key)
        if result.is_ok():
            settings = result.ok()
            # check if the settings list is empty
            if len(settings) == 0:
                return SyftError(message="No settings found")
            result = settings[0]
            return Ok(result)
        else:
            return SyftError(message=result.err())

    @service_method(path="settings.set", name="set")
    def set(
        self, context: AuthedServiceContext, settings: NodeSettings
    ) -> Result[Ok, Err]:
        """Set a new the Node Settings"""
        result = self.stash.set(context.credentials, settings)
        if result.is_ok():
            return result
        else:
            return SyftError(message=result.err())

    @service_method(path="settings.update", name="update", autosplat=["settings"])
    def update(
        self, context: AuthedServiceContext, settings: NodeSettingsUpdate
    ) -> Result[SyftSuccess, SyftError]:
        res = self._update(context, settings)
        if res.is_ok():
            return SyftSuccess(
                message=(
                    "Settings updated successfully. "
                    + "You must call <client>.refresh() to sync your client with the changes."
                )
            )
        else:
            return SyftError(message=res.err())

    def _update(
        self, context: AuthedServiceContext, settings: NodeSettingsUpdate
    ) -> Result[Ok, Err]:
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
            Result[SyftSuccess, SyftError]: A result indicating the success or failure of the update operation.

        Example:
        >>> node_client.update(name='foo', organization='bar', description='baz', signup_enabled=True)
        SyftSuccess: Settings updated successfully.
        """
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            current_settings = result.ok()
            if len(current_settings) > 0:
                new_settings = current_settings[0].model_copy(
                    update=settings.to_dict(exclude_empty=True)
                )
                update_result = self.stash.update(context.credentials, new_settings)
                return update_result
            else:
                return Err(value="No settings found")
        else:
            return result

    @service_method(
        path="settings.set_node_side_type_dangerous",
        name="set_node_side_type_dangerous",
        roles=ADMIN_ROLE_LEVEL,
    )
    def set_node_side_type_dangerous(
        self, context: AuthedServiceContext, node_side_type: str
    ) -> Result[SyftSuccess, SyftError]:
        side_type_options = [e.value for e in NodeSideType]
        if node_side_type not in side_type_options:
            return SyftError(
                message=f"Not a valid node_side_type, please use one of the options from: {side_type_options}"
            )

        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            current_settings = result.ok()
            if len(current_settings) > 0:
                new_settings = current_settings[0]
                new_settings.node_side_type = node_side_type
                update_result = self.stash.update(context.credentials, new_settings)
                if update_result.is_ok():
                    return SyftSuccess(
                        message=(
                            "Settings updated successfully. "
                            + "You must call <client>.refresh() to sync your client with the changes."
                        )
                    )
                else:
                    return SyftError(message=update_result.err())
            else:
                return SyftError(message="No settings found")
        else:
            return SyftError(message=result.err())

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

        settings = NodeSettingsUpdate(signup_enabled=enable)
        result = self._update(context=context, settings=settings)

        if isinstance(result, SyftError):
            return SyftError(message=f"Failed to update settings: {result.err()}")

        message = "enabled" if enable else "disabled"
        return SyftSuccess(message=f"Registration feature successfully {message}")

    @service_method(
        path="settings.enable_eager_execution",
        name="enable_eager_execution",
        roles=ADMIN_ROLE_LEVEL,
        warning=HighSideCRUDWarning(confirmation=True),
    )
    def enable_eager_execution(
        self, context: AuthedServiceContext, enable: bool
    ) -> SyftSuccess | SyftError:
        """Enable/Disable eager execution."""
        settings = NodeSettingsUpdate(eager_execution_enabled=enable)

        result = self._update(context=context, settings=settings)

        if result.is_err():
            return SyftError(message=f"Failed to update settings: {result.err()}")

        message = "enabled" if enable else "disabled"
        return SyftSuccess(message=f"Eager execution {message}")

    @service_method(
        path="settings.allow_association_request_auto_approval",
        name="allow_association_request_auto_approval",
    )
    def allow_association_request_auto_approval(
        self, context: AuthedServiceContext, enable: bool
    ) -> SyftSuccess | SyftError:
        new_settings = NodeSettingsUpdate(association_request_auto_approval=enable)
        result = self._update(context, settings=new_settings)
        if isinstance(result, SyftError):
            return result

        message = "enabled" if enable else "disabled"
        return SyftSuccess(
            message="Association request auto-approval successfully " + message
        )

    @service_method(
        path="settings.welcome_preview",
        name="welcome_preview",
    )
    def welcome_preview(
        self,
        context: AuthedServiceContext,
        markdown: str = "",
        html: str = "",
    ) -> MarkdownDescription | HTMLObject | SyftError:
        if not markdown and not html or markdown and html:
            return SyftError(
                message="Invalid markdown/html fields. You must set one of them."
            )

        welcome_msg = None
        if markdown:
            welcome_msg = MarkdownDescription(text=markdown)
        else:
            welcome_msg = HTMLObject(text=html)

        return welcome_msg

    @service_method(
        path="settings.welcome_customize",
        name="welcome_customize",
    )
    def welcome_customize(
        self,
        context: AuthedServiceContext,
        markdown: str = "",
        html: str = "",
    ) -> SyftSuccess | SyftError:
        if not markdown and not html or markdown and html:
            return SyftError(
                message="Invalid markdown/html fields. You must set one of them."
            )

        welcome_msg = None
        if markdown:
            welcome_msg = MarkdownDescription(text=markdown)
        else:
            welcome_msg = HTMLObject(text=html)

        new_settings = NodeSettingsUpdate(welcome_markdown=welcome_msg)
        result = self._update(context=context, settings=new_settings)
        if isinstance(result, SyftError):
            return result

        return SyftSuccess(message="Welcome Markdown was successfully updated!")

    @service_method(
        path="settings.welcome_show",
        name="welcome_show",
        roles=GUEST_ROLE_LEVEL,
    )
    def welcome_show(
        self,
        context: AuthedServiceContext,
    ) -> HTMLObject | MarkdownDescription | SyftError:
        result = self.stash.get_all(context.node.signing_key.verify_key)
        user_service = context.node.get_service("userservice")
        role = user_service.get_role_for_credentials(context.credentials)
        if result.is_ok():
            settings = result.ok()
            # check if the settings list is empty
            if len(settings) == 0:
                return SyftError(message="No settings found")
            settings = settings[0]
            if settings.welcome_markdown:
                str_tmp = Template(settings.welcome_markdown.text)
                welcome_msg_class = type(settings.welcome_markdown)
                node_side_type = (
                    "Low Side"
                    if context.node.metadata.node_side_type
                    == NodeSideType.LOW_SIDE.value
                    else "High Side"
                )
                commands = ""
                if (
                    role.value == ServiceRole.NONE.value
                    or role.value == ServiceRole.GUEST.value
                ):
                    commands = GUEST_COMMANDS
                elif (
                    role is not None and role.value == ServiceRole.DATA_SCIENTIST.value
                ):
                    commands = DS_COMMANDS
                elif role is not None and role.value >= ServiceRole.DATA_OWNER.value:
                    commands = DO_COMMANDS

                command_list = f"""
                <ul style='padding-left: 1em;'>
                    {commands}
                </ul>
                """
                result = str_tmp.safe_substitute(
                    FONT_CSS=FONT_CSS,
                    grid_symbol=load_png_base64("small-grid-symbol-logo.png"),
                    domain_name=context.node.name,
                    # node_url='http://testing:8080',
                    node_type=context.node.metadata.node_type.capitalize(),
                    node_side_type=node_side_type,
                    node_version=context.node.metadata.syft_version,
                    command_list=command_list,
                )
                return welcome_msg_class(text=result)
            return SyftError(message="There's no welcome message")
        else:
            return SyftError(message=result.err())
