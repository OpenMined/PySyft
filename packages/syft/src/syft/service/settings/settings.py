# stdlib
import logging
from typing import Any

# relative
from ...abstract_node import NodeSideType
from ...abstract_node import NodeType
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...service.worker.utils import DEFAULT_WORKER_POOL_NAME
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.misc_objs import HTMLObject
from ...util.misc_objs import MarkdownDescription
from ...util.schema import DEFAULT_WELCOME_MSG

logger = logging.getLogger(__name__)


@serializable()
class NodeSettingsUpdate(PartialSyftObject):
    __canonical_name__ = "NodeSettingsUpdate"
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
class NodeSettings(SyftObject):
    __canonical_name__ = "NodeSettings"
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
    name: str = "Node"
    deployed_on: str
    organization: str = "OpenMined"
    verify_key: SyftVerifyKey
    on_board: bool = True
    description: str = "This is the default description for a Domain Node."
    node_type: NodeType = NodeType.DOMAIN
    signup_enabled: bool
    admin_email: str
    node_side_type: NodeSideType = NodeSideType.HIGH_SIDE
    show_warnings: bool
    association_request_auto_approval: bool
    eager_execution_enabled: bool = False
    default_worker_pool: str = DEFAULT_WORKER_POOL_NAME
    welcome_markdown: HTMLObject | MarkdownDescription = HTMLObject(
        text=DEFAULT_WELCOME_MSG
    )

    def _repr_html_(self) -> Any:
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
                <p><strong>Admin email: </strong>{self.admin_email}</p>
            </div>

            """
