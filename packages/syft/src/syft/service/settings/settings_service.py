# stdlib
from string import Template
from typing import Any

# third party
from pydantic import ValidationError

# relative
from ...abstract_server import ServerSideType
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_metaclass import Empty
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
from ..notifier.notifier_enums import EMAIL_TYPES
from ..notifier.notifier_enums import NOTIFICATION_FREQUENCY
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..user.user_roles import ServiceRole
from ..warnings import HighSideCRUDWarning
from .settings import ServerSettings
from .settings import ServerSettingsUpdate
from .settings_stash import SettingsStash

# for testing purpose
_NOTIFICATIONS_ENABLED_WIHOUT_CREDENTIALS_ERROR = (
    "Failed to enable notification. "
    "Email credentials are invalid or have not been set. "
    "Please use `enable_notifications` from `user_service` "
    "to set the correct email credentials."
)


@serializable(canonical_name="SettingsService", version=1)
class SettingsService(AbstractService):
    stash: SettingsStash

    def __init__(self, store: DBManager) -> None:
        self.stash = SettingsStash(store=store)

    @service_method(path="settings.get", name="get")
    def get(self, context: UnauthedServiceContext) -> ServerSettings:
        """Get Settings"""
        all_settings = self.stash.get_all(
            context.server.signing_key.verify_key
        ).unwrap()

        if len(all_settings) == 0:
            raise NotFoundException(public_message="No settings found")

        return all_settings[0]

    @service_method(path="settings.set", name="set")
    def set(
        self, context: AuthedServiceContext, settings: ServerSettings
    ) -> ServerSettings:
        """Set a new the Server Settings"""
        return self.stash.set(context.credentials, settings).unwrap()

    @service_method(
        path="settings.update",
        name="update",
        autosplat=["settings"],
        unwrap_on_success=False,
        roles=ADMIN_ROLE_LEVEL,
    )
    def update(
        self, context: AuthedServiceContext, settings: ServerSettingsUpdate
    ) -> SyftSuccess:
        """
        Update the Server Settings using the provided values.

        Args:
            name: Optional[str]
                Server name
            organization: Optional[str]
                Organization name
            description: Optional[str]
                Server description
            on_board: Optional[bool]
                Show onboarding panel when a user logs in for the first time
            signup_enabled: Optional[bool]
                Enable/Disable registration
            admin_email: Optional[str]
                Administrator email
            association_request_auto_approval: Optional[bool]

        Returns:
            SyftSuccess: Message indicating the success of the operation, with the
                update server settings as the value property.

        Example:
        >>> server_client.update(name='foo', organization='bar', description='baz', signup_enabled=True)
        SyftSuccess: Settings updated successfully.
        """
        updated_settings = self._update(context, settings).unwrap()
        return SyftSuccess(
            message=(
                "Settings updated successfully. "
                + "You must call <client>.refresh() to sync your client with the changes."
            ),
            value=updated_settings,
        )

    @as_result(StashException, NotFoundException, ValidationError)
    def _update(
        self, context: AuthedServiceContext, settings: ServerSettingsUpdate
    ) -> ServerSettings:
        all_settings = self.stash.get_all(
            context.credentials, limit=1, sort_order="desc"
        ).unwrap()
        if len(all_settings) > 0:
            new_settings = all_settings[0].model_copy(
                update=settings.to_dict(exclude_empty=True)
            )
            ServerSettings.model_validate(new_settings.to_dict())
            update_result = self.stash.update(
                context.credentials, obj=new_settings
            ).unwrap()

            # If notifications_enabled is present in the update, we need to update the notifier settings
            if settings.notifications_enabled is not Empty:  # type: ignore[comparison-overlap]
                notifier_settings_res = context.server.services.notifier.settings(
                    context
                )
                if (
                    not notifier_settings_res.is_ok()
                    or notifier_settings_res.ok() is None
                ):
                    raise SyftException(
                        public_message=(
                            "Notification has not been enabled. "
                            "Please use `enable_notifications` from `user_service`."
                        )
                    )

                context.server.services.notifier._set_notifier(
                    context, active=settings.notifications_enabled
                )

            return update_result
        else:
            raise NotFoundException(public_message="Server settings not found")

    @service_method(
        path="settings.set_server_side_type_dangerous",
        name="set_server_side_type_dangerous",
        roles=ADMIN_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def set_server_side_type_dangerous(
        self, context: AuthedServiceContext, server_side_type: str
    ) -> SyftSuccess:
        side_type_options = [e.value for e in ServerSideType]

        if server_side_type not in side_type_options:
            raise SyftException(
                public_message=f"Not a valid server_side_type, please use one of the options from: {side_type_options}"
            )

        current_settings = self.stash.get_all(
            context.credentials, limit=1, sort_order="desc"
        ).unwrap()
        if len(current_settings) > 0:
            new_settings = current_settings[0]
            new_settings.server_side_type = ServerSideType(server_side_type)
            updated_settings = self.stash.update(
                context.credentials, new_settings
            ).unwrap()
            return SyftSuccess(
                message=(
                    "Settings updated successfully. "
                    + "You must call <client>.refresh() to sync your client with the changes."
                ),
                value=updated_settings,
            )
        else:
            # TODO: Turn this into a function?
            raise NotFoundException(public_message="Server settings not found")

    @service_method(
        path="settings.batch_notifications",
        name="batch_notifications",
        roles=ADMIN_ROLE_LEVEL,
    )
    def batch_notifications(
        self,
        context: AuthedServiceContext,
        email_type: EMAIL_TYPES,
        frequency: NOTIFICATION_FREQUENCY,
        start_time: str = "",
    ) -> SyftSuccess:
        result = context.server.services.notifier.set_email_batch(
            context=context,
            email_type=email_type,
            frequency=frequency,
            start_time=start_time,
        ).unwrap()
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
    ) -> SyftSuccess:
        context.server.services.notifier.turn_on(
            context=context,
            email_username=email_username,
            email_password=email_password,
            email_sender=email_sender,
            email_server=email_server,
            email_port=email_port,
        ).unwrap()
        return SyftSuccess(message="Notifications enabled")

    @service_method(
        path="settings.disable_notifications",
        name="disable_notifications",
        roles=ADMIN_ROLE_LEVEL,
    )
    def disable_notifications(
        self,
        context: AuthedServiceContext,
    ) -> SyftSuccess:
        context.server.services.notifier.turn_off(context=context).unwrap()
        return SyftSuccess(message="Notifications disabled")

    @service_method(
        path="settings.allow_guest_signup",
        name="allow_guest_signup",
        warning=HighSideCRUDWarning(confirmation=True),
        unwrap_on_success=False,
    )
    def allow_guest_signup(
        self, context: AuthedServiceContext, enable: bool
    ) -> SyftSuccess:
        """Enable/Disable Registration for Data Scientist or Guest Users."""
        flags.CAN_REGISTER = enable

        settings = ServerSettingsUpdate(signup_enabled=enable)
        self._update(context=context, settings=settings).unwrap()
        message = "enabled" if enable else "disabled"
        return SyftSuccess(
            message=f"Registration feature successfully {message}", value=message
        )

    # NOTE: This service is disabled until we bring back Eager Execution
    # @service_method(
    #     path="settings.enable_eager_execution",
    #     name="enable_eager_execution",
    #     roles=ADMIN_ROLE_LEVEL,
    #     warning=HighSideCRUDWarning(confirmation=True),
    # )
    def enable_eager_execution(
        self, context: AuthedServiceContext, enable: bool
    ) -> SyftSuccess:
        """Enable/Disable eager execution."""
        settings = ServerSettingsUpdate(eager_execution_enabled=enable)
        self._update(context=context, settings=settings).unwrap()
        message = "enabled" if enable else "disabled"
        return SyftSuccess(message=f"Eager execution {message}", value=message)

    @service_method(path="settings.set_email_rate_limit", name="set_email_rate_limit")
    def set_email_rate_limit(
        self, context: AuthedServiceContext, email_type: EMAIL_TYPES, daily_limit: int
    ) -> SyftSuccess:
        return context.server.services.notifier.set_email_rate_limit(
            context, email_type, daily_limit
        )

    @service_method(
        path="settings.allow_association_request_auto_approval",
        name="allow_association_request_auto_approval",
        unwrap_on_success=False,
    )
    def allow_association_request_auto_approval(
        self, context: AuthedServiceContext, enable: bool
    ) -> SyftSuccess:
        new_settings = ServerSettingsUpdate(association_request_auto_approval=enable)
        self._update(context, settings=new_settings).unwrap()
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
    ) -> MarkdownDescription | HTMLObject:
        if not markdown and not html or markdown and html:
            raise SyftException(
                public_message="Invalid markdown/html fields. You must set one of them."
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
        unwrap_on_success=False,
    )
    def welcome_customize(
        self,
        context: AuthedServiceContext,
        markdown: str = "",
        html: str = "",
    ) -> SyftSuccess:
        if not markdown and not html or markdown and html:
            raise SyftException(
                public_message="Invalid markdown/html fields. You must set one of them."
            )

        welcome_msg = None
        if markdown:
            welcome_msg = MarkdownDescription(text=markdown)
        else:
            welcome_msg = HTMLObject(text=html)

        new_settings = ServerSettingsUpdate(welcome_markdown=welcome_msg)
        self._update(context=context, settings=new_settings).unwrap()

        return SyftSuccess(message="Welcome Markdown was successfully updated!")

    @service_method(
        path="settings.welcome_show",
        name="welcome_show",
        roles=GUEST_ROLE_LEVEL,
    )
    def welcome_show(
        self,
        context: AuthedServiceContext,
    ) -> HTMLObject | MarkdownDescription:
        all_settings = self.stash.get_all(
            context.server.signing_key.verify_key
        ).unwrap()
        role = context.server.services.user.get_role_for_credentials(
            context.credentials
        ).unwrap()

        # check if the settings list is empty
        if len(all_settings) == 0:
            raise NotFoundException(public_message="Server settings not found")
        settings = all_settings[0]

        if settings.welcome_markdown:
            str_tmp = Template(settings.welcome_markdown.text)
            welcome_msg_class = type(settings.welcome_markdown)
            server_side_type = (
                "Low Side"
                if context.server.metadata.server_side_type
                == ServerSideType.LOW_SIDE.value
                else "High Side"
            )
            commands = ""
            if (
                role.value == ServiceRole.NONE.value
                or role.value == ServiceRole.GUEST.value
            ):
                commands = GUEST_COMMANDS
            elif role is not None and role.value == ServiceRole.DATA_SCIENTIST.value:
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
                server_symbol=load_png_base64("small-syft-symbol-logo.png"),
                datasite_name=context.server.name,
                description=context.server.metadata.description,
                # server_url='http://testing:8080',
                server_type=context.server.metadata.server_type.capitalize(),
                server_side_type=server_side_type,
                server_version=context.server.metadata.syft_version,
                command_list=command_list,
            )
            return welcome_msg_class(text=result)
        raise SyftException(public_message="There's no welcome message")

    @service_method(
        path="settings.get_server_config",
        name="get_server_config",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_server_config(
        self,
        context: AuthedServiceContext,
    ) -> dict[str, Any]:
        server = context.server

        return {
            "name": server.name,
            "server_type": server.server_type,
            # "deploy_to": server.deployment_type_enum,
            "server_side_type": server.server_side_type,
            # "port": server.port,
            "processes": server.processes,
            "dev_mode": server.dev_mode,
            "reset": True,  # we should be able to get all the objects from migration data
            "tail": False,
            # "host": server.host,
            "enable_warnings": server.enable_warnings,
            "n_consumers": server.queue_config.client_config.create_producer,
            "thread_workers": server.queue_config.thread_workers,
            "create_producer": server.queue_config.client_config.create_producer,
            "queue_port": server.queue_config.client_config.queue_port,
            "association_request_auto_approval": server.association_request_auto_approval,
            "background_tasks": True,
            "debug": True,  # we also want to debug
            "migrate": False,  # I think we dont want to migrate?
        }
