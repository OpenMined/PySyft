# stdlib
from collections.abc import Callable
import logging
from typing import Any

# relative
from ...abstract_server import ServerSideType
from ...abstract_server import ServerType
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...service.worker.utils import DEFAULT_WORKER_POOL_NAME
from ...types.syft_migration import migrate
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.transforms import drop
from ...types.transforms import make_set_default
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.misc_objs import HTMLObject
from ...util.misc_objs import MarkdownDescription
from ...util.schema import DEFAULT_WELCOME_MSG

logger = logging.getLogger(__name__)


@serializable()
class ServerSettingsUpdateV1(PartialSyftObject):
    __canonical_name__ = "ServerSettingsUpdate"
    __version__ = SYFT_OBJECT_VERSION_1
    id: UID
    name: str
    organization: str
    description: str
    on_board: bool
    signup_enabled: bool
    admin_email: str
    association_request_auto_approval: bool
    welcome_markdown: HTMLObject | MarkdownDescription
    eager_execution_enabled: bool = False


@serializable()
class ServerSettingsUpdate(PartialSyftObject):
    __canonical_name__ = "ServerSettingsUpdate"
    __version__ = SYFT_OBJECT_VERSION_2
    id: UID
    name: str
    organization: str
    description: str
    on_board: bool
    signup_enabled: bool
    admin_email: str
    association_request_auto_approval: bool
    welcome_markdown: HTMLObject | MarkdownDescription
    eager_execution_enabled: bool
    notifications_enabled: bool


@serializable()
class ServerSettingsV1(SyftObject):
    __canonical_name__ = "ServerSettings"
    __version__ = SYFT_OBJECT_VERSION_1
    __repr_attrs__ = [
        "name",
        "organization",
        "description",
        "deployed_on",
        "signup_enabled",
        "admin_email",
    ]

    id: UID
    name: str = "Server"
    deployed_on: str
    organization: str = "OpenMined"
    verify_key: SyftVerifyKey
    on_board: bool = True
    description: str = "This is the default description for a Datasite Server."
    server_type: ServerType = ServerType.DATASITE
    signup_enabled: bool
    admin_email: str
    server_side_type: ServerSideType = ServerSideType.HIGH_SIDE
    show_warnings: bool
    association_request_auto_approval: bool
    eager_execution_enabled: bool = False
    default_worker_pool: str = DEFAULT_WORKER_POOL_NAME
    welcome_markdown: HTMLObject | MarkdownDescription = HTMLObject(
        text=DEFAULT_WELCOME_MSG
    )


@serializable()
class ServerSettings(SyftObject):
    __canonical_name__ = "ServerSettings"
    __version__ = SYFT_OBJECT_VERSION_2
    __repr_attrs__ = [
        "name",
        "organization",
        "description",
        "deployed_on",
        "signup_enabled",
        "admin_email",
    ]

    id: UID
    name: str = "Server"
    deployed_on: str
    organization: str = "OpenMined"
    verify_key: SyftVerifyKey
    on_board: bool = True
    description: str = "This is the default description for a Datasite Server."
    server_type: ServerType = ServerType.DATASITE
    signup_enabled: bool
    admin_email: str
    server_side_type: ServerSideType = ServerSideType.HIGH_SIDE
    show_warnings: bool
    association_request_auto_approval: bool
    eager_execution_enabled: bool = False
    default_worker_pool: str = DEFAULT_WORKER_POOL_NAME
    welcome_markdown: HTMLObject | MarkdownDescription = HTMLObject(
        text=DEFAULT_WELCOME_MSG
    )
    notifications_enabled: bool

    def _repr_html_(self) -> Any:
        # .api.services.notifications.settings() is how the server itself would dispatch notifications.
        # .api.services.notifications.user_settings() sets if a specific user wants or not to receive notifications.
        # Class NotifierSettings holds both pieces of info.
        # Users will get notification x where x in {email, slack, sms, app} if three things are set to True:
        # 1) .....settings().active
        # 2) .....settings().x_enabled
        # 3) .....user_settings().x

        preferences = self._get_api().services.notifications.settings()
        if not preferences:
            notification_print_str = "Create notification settings using enable_notifications from user_service"
        else:
            notifications = []
            if preferences.email_enabled:
                notifications.append("email")
            if preferences.sms_enabled:
                notifications.append("sms")
            if preferences.slack_enabled:
                notifications.append("slack")
            if preferences.app_enabled:
                notifications.append("app")

            # self.notifications_enabled = preferences.active
            if preferences.active:
                if notifications:
                    notification_print_str = f"Enabled via {', '.join(notifications)}"
                else:
                    notification_print_str = "Enabled without any communication method"
            else:
                notification_print_str = "Disabled"

        return f"""
            <style>
            .syft-settings {{color: {SURFACE[options.color_theme]};}}
            </style>
            <div class='syft-settings'>
                <h3>Settings</h3>
                <p><strong>Id: </strong>{self.id}</p>
                <p><strong>Name: </strong>{self.name}</p>
                <p><strong>Organization: </strong>{self.organization}</p>
                <p><strong>Description: </strong>{self.description}</p>
                <p><strong>Deployed on: </strong>{self.deployed_on}</p>
                <p><strong>Signup enabled: </strong>{self.signup_enabled}</p>
                <p><strong>Notifications enabled: </strong>{notification_print_str}</p>
                <p><strong>Admin email: </strong>{self.admin_email}</p>
            </div>

            """


@migrate(ServerSettingsV1, ServerSettings)
def migrate_server_settings_v1_to_v2() -> list[Callable]:
    return [make_set_default("notifications_enabled", False)]


@migrate(ServerSettings, ServerSettingsV1)
def migrate_server_settings_v2_to_v1() -> list[Callable]:
    # Use drop function on "notifications_enabled" attrubute
    return [drop(["notifications_enabled"])]


@migrate(ServerSettingsUpdateV1, ServerSettingsUpdate)
def migrate_server_settings_update_v1_to_v2() -> list[Callable]:
    return [make_set_default("notifications_enabled", False)]


@migrate(ServerSettingsUpdate, ServerSettingsUpdateV1)
def migrate_server_settings_update_v2_to_v1() -> list[Callable]:
    return [drop(["notifications_enabled"])]
