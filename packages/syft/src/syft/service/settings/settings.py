# stdlib
import logging
from typing import Any

# third party
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

# relative
from ...abstract_server import ServerSideType
from ...abstract_server import ServerType
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...service.worker.utils import DEFAULT_WORKER_POOL_NAME
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.misc_objs import HTMLObject
from ...util.misc_objs import MarkdownDescription
from ...util.schema import DEFAULT_WELCOME_MSG

logger = logging.getLogger(__name__)


@serializable()
class PwdTokenResetConfig(SyftObject):
    __canonical_name__ = "PwdTokenResetConfig"
    __version__ = SYFT_OBJECT_VERSION_1
    ascii: bool = True
    numbers: bool = True
    token_len: int = 12

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
    eager_execution_enabled: bool = False
    pwd_token_config: PwdTokenResetConfig


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
    pwd_token_config: PwdTokenResetConfig = PwdTokenResetConfig()
    welcome_markdown: HTMLObject | MarkdownDescription = HTMLObject(
        text=DEFAULT_WELCOME_MSG
    )

    def _repr_html_(self) -> Any:
        preferences = self._get_api().services.notifications.user_settings()
        notifications = []
        if preferences.email:
            notifications.append("email")
        if preferences.sms:
            notifications.append("sms")
        if preferences.slack:
            notifications.append("slack")
        if preferences.app:
            notifications.append("app")

        if notifications:
            notifications_enabled = f"True via {', '.join(notifications)}"
        else:
            notifications_enabled = "False"

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
                <p><strong>Notifications enabled: </strong>{notifications_enabled}</p>
                <p><strong>Admin email: </strong>{self.admin_email}</p>
            </div>

            """
