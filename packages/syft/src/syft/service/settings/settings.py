# stdlib
from collections.abc import Callable
import logging
from typing import Any

# third party
from pydantic import EmailStr
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

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
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import SYFT_OBJECT_VERSION_4
from ...types.syft_object import SyftObject
from ...types.transforms import drop
from ...types.transforms import make_set_default
from ...types.uid import UID
from ...util.misc_objs import HTMLObject
from ...util.misc_objs import MarkdownDescription
from ...util.schema import DEFAULT_WELCOME_MSG

logger = logging.getLogger(__name__)

MIN_ORG_NAME_LENGTH = 1
MIN_SERVER_NAME_LENGTH = 1


@serializable()
class PwdTokenResetConfig(SyftObject):
    __canonical_name__ = "PwdTokenResetConfig"
    __version__ = SYFT_OBJECT_VERSION_1

    ascii: bool = True
    numbers: bool = True
    token_len: int = 12
    # Token expiration time in seconds (not minutes)
    token_exp_min: int = 1800  # TODO: Rename variable to token_exp_sec

    @model_validator(mode="after")
    def validate_char_types(self) -> Self:
        if not self.ascii and not self.numbers:
            raise ValueError(
                "Invalid config, at least one of the ascii/number options must be true."
            )

        return self

    @field_validator("token_len")
    @classmethod
    def check_token_len(cls, value: int) -> int:
        if value < 4:
            raise ValueError("Token length must be greater than 4.")
        return value


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
class ServerSettingsUpdateV2(PartialSyftObject):
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
class ServerSettingsUpdateV3(PartialSyftObject):
    __canonical_name__ = "ServerSettingsUpdate"
    __version__ = SYFT_OBJECT_VERSION_3
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
    pwd_token_config: PwdTokenResetConfig


@serializable()
class ServerSettingsUpdate(PartialSyftObject):
    __canonical_name__ = "ServerSettingsUpdate"
    __version__ = SYFT_OBJECT_VERSION_4
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
    pwd_token_config: PwdTokenResetConfig
    allow_guest_sessions: bool


@serializable()
class ServerSettingsV1(SyftObject):
    __canonical_name__ = "ServerSettings"
    __version__ = SYFT_OBJECT_VERSION_1

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
class ServerSettingsV2(SyftObject):
    __canonical_name__ = "ServerSettings"
    __version__ = SYFT_OBJECT_VERSION_2

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


@serializable()
class ServerSettingsV3(SyftObject):
    __canonical_name__ = "ServerSettings"
    __version__ = SYFT_OBJECT_VERSION_3
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
    pwd_token_config: PwdTokenResetConfig = PwdTokenResetConfig()


@serializable()
class ServerSettings(SyftObject):
    __canonical_name__ = "ServerSettings"
    __version__ = SYFT_OBJECT_VERSION_4
    __repr_attrs__ = [
        "name",
        "organization",
        "description",
        "deployed_on",
        "signup_enabled",
        "admin_email",
        "allow_guest_sessions",
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
    admin_email: EmailStr
    server_side_type: ServerSideType = ServerSideType.HIGH_SIDE
    show_warnings: bool
    association_request_auto_approval: bool
    eager_execution_enabled: bool = False
    default_worker_pool: str = DEFAULT_WORKER_POOL_NAME
    welcome_markdown: HTMLObject | MarkdownDescription = HTMLObject(
        text=DEFAULT_WELCOME_MSG
    )
    notifications_enabled: bool
    pwd_token_config: PwdTokenResetConfig = PwdTokenResetConfig()
    allow_guest_sessions: bool = True

    @field_validator("organization")
    def organization_length(cls, v: str) -> str:
        if len(v) < MIN_ORG_NAME_LENGTH:
            raise ValueError(
                f"'organization' must be at least {MIN_ORG_NAME_LENGTH} characters long"
            )
        return v

    @field_validator("name")
    def name_length(cls, v: str) -> str:
        if len(v) < MIN_SERVER_NAME_LENGTH:
            raise ValueError(
                f'"name" must be at least {MIN_SERVER_NAME_LENGTH} characters long'
            )
        return v

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
                <p><strong>Enable guest sessions: </strong>{self.allow_guest_sessions}</p>
            </div>

            """


# Server Settings Migration


# set
@migrate(ServerSettingsV1, ServerSettingsV2)
def migrate_server_settings_v1_to_v2() -> list[Callable]:
    return [make_set_default("notifications_enabled", False)]


@migrate(ServerSettingsV2, ServerSettingsV3)
def migrate_server_settings_v2_to_v3() -> list[Callable]:
    return [make_set_default("pwd_token_config", PwdTokenResetConfig())]


@migrate(ServerSettingsV3, ServerSettings)
def migrate_server_settings_v3_to_current() -> list[Callable]:
    return [make_set_default("allow_guest_sessions", False)]


# drop
@migrate(ServerSettingsV2, ServerSettingsV1)
def migrate_server_settings_v2_to_v1() -> list[Callable]:
    # Use drop function on "notifications_enabled" attrubute
    return [drop(["notifications_enabled"])]


@migrate(ServerSettingsV3, ServerSettingsV2)
def migrate_server_settings_v3_to_v2() -> list[Callable]:
    # Use drop function on "notifications_enabled" attrubute
    return [drop(["pwd_token_config"])]


@migrate(ServerSettings, ServerSettingsV3)
def migrate_server_settings_current_to_v3() -> list[Callable]:
    # Use drop function on "notifications_enabled" attrubute
    return [drop(["allow_guest_sessions"])]


# Server Settings Update Migration


# set
@migrate(ServerSettingsUpdateV1, ServerSettingsUpdateV2)
def migrate_server_settings_update_v1_to_v2() -> list[Callable]:
    return [make_set_default("notifications_enabled", False)]


@migrate(ServerSettingsUpdateV2, ServerSettingsUpdateV3)
def migrate_server_settings_update_v2_to_v3() -> list[Callable]:
    return [make_set_default("pwd_token_config", PwdTokenResetConfig())]


@migrate(ServerSettingsUpdateV3, ServerSettingsUpdate)
def migrate_server_settings_update_v3_to_current() -> list[Callable]:
    return [make_set_default("allow_guest_sessions", False)]


# drop
@migrate(ServerSettingsUpdateV2, ServerSettingsUpdateV1)
def migrate_server_settings_update_v2_to_v1() -> list[Callable]:
    return [drop(["notifications_enabled"])]


@migrate(ServerSettingsUpdateV3, ServerSettingsUpdateV2)
def migrate_server_settings_update_current_to_v2() -> list[Callable]:
    return [drop(["pwd_token_config"])]


@migrate(ServerSettingsUpdate, ServerSettingsUpdateV3)
def migrate_server_settings_update_current_to_v3() -> list[Callable]:
    return [drop(["allow_guest_sessions"])]
