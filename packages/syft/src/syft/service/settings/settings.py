# stdlib
from collections.abc import Callable
import logging
from typing import Any

# third party
from IPython.display import display
from pydantic import field_validator

# relative
from ...abstract_server import ServerSideType
from ...abstract_server import ServerType
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...service.worker.utils import DEFAULT_WORKER_POOL_NAME
from ...types.syft_metaclass import Empty
from ...types.syft_migration import migrate
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import SYFT_OBJECT_VERSION_4
from ...types.syft_object import SYFT_OBJECT_VERSION_5
from ...types.syft_object import SYFT_OBJECT_VERSION_6
from ...types.syft_object import SyftObject
from ...types.transforms import make_set_default
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.misc_objs import HTMLObject
from ...util.misc_objs import MarkdownDescription
from ...util.schema import DEFAULT_WELCOME_MSG
from ...util.util import get_env
from ..response import SyftInfo

logger = logging.getLogger(__name__)


@serializable()
class ServerSettingsUpdateV4(PartialSyftObject):
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
    server_side_type: str

    @field_validator("server_side_type", check_fields=False)
    @classmethod
    def validate_server_side_type(cls, v: str) -> type[Empty]:
        msg = f"You cannot update 'server_side_type' through ServerSettingsUpdate. \
Please use client.set_server_side_type_dangerous(server_side_type={v}). \
Be aware if you have private data on the server and you want to change it to the Low Side, \
as information might be leaked."
        try:
            display(SyftInfo(message=msg))
        except Exception as e:
            logger.error(msg, exc_info=e)
        return Empty


@serializable()
class ServerSettingsUpdate(PartialSyftObject):
    __canonical_name__ = "ServerSettingsUpdate"
    __version__ = SYFT_OBJECT_VERSION_5
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
class ServerSettings(SyftObject):
    __canonical_name__ = "ServerSettings"
    __version__ = SYFT_OBJECT_VERSION_6
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


@serializable()
class ServerSettingsV5(SyftObject):
    __canonical_name__ = "ServerSettings"
    __version__ = SYFT_OBJECT_VERSION_5
    __repr_attrs__ = [
        "name",
        "organization",
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
    description: str = "Text"
    server_type: ServerType = ServerType.DATASITE
    signup_enabled: bool
    admin_email: str
    server_side_type: ServerSideType = ServerSideType.HIGH_SIDE
    show_warnings: bool
    association_request_auto_approval: bool
    default_worker_pool: str = DEFAULT_WORKER_POOL_NAME
    welcome_markdown: HTMLObject | MarkdownDescription = HTMLObject(
        text=DEFAULT_WELCOME_MSG
    )


@serializable()
class ServerSettingsV2(SyftObject):
    __canonical_name__ = "ServerSettings"
    __version__ = SYFT_OBJECT_VERSION_3
    __repr_attrs__ = [
        "name",
        "organization",
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
    description: str = "Text"
    server_type: ServerType = ServerType.DATASITE
    signup_enabled: bool
    admin_email: str
    server_side_type: ServerSideType = ServerSideType.HIGH_SIDE
    show_warnings: bool


# @migrate(ServerSettingsV3, ServerSettingsV5)
# def upgrade_server_settings() -> list[Callable]:
#     return [
#         make_set_default("association_request_auto_approval", False),
#         make_set_default(
#             "default_worker_pool",
#             get_env("DEFAULT_WORKER_POOL_NAME", DEFAULT_WORKER_POOL_NAME),
#         ),
#     ]

# @migrate(ServerSettingsV3, ServerSettings)
# def upgrade_server_settings_v3_to_v6() -> list[Callable]:
#     return [
#         make_set_default("association_request_auto_approval", False),
#         make_set_default(
#             "default_worker_pool",
#             get_env("DEFAULT_WORKER_POOL_NAME", DEFAULT_WORKER_POOL_NAME),
#         ),
#         make_set_default("eager_execution_enabled", False),
#     ]


@migrate(ServerSettingsV2, ServerSettings)
def upgrade_server_settings() -> list[Callable]:
    return [
        make_set_default("association_request_auto_approval", False),
        make_set_default(
            "default_worker_pool",
            get_env("DEFAULT_WORKER_POOL_NAME", DEFAULT_WORKER_POOL_NAME),
        ),
        make_set_default("eager_execution_enabled", False),
    ]
