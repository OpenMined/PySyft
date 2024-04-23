# stdlib
from typing import Callable

# relative
from ...abstract_node import NodeSideType
from ...abstract_node import NodeType
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...service.worker.utils import DEFAULT_WORKER_POOL_NAME
from ...types.syft_migration import migrate
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import SYFT_OBJECT_VERSION_4
from ...types.syft_object import SyftObject
from ...types.transforms import drop
from ...types.transforms import make_set_default
from ...types.uid import UID


@serializable()
class NodeSettingsUpdateV2(PartialSyftObject):
    __canonical_name__ = "NodeSettingsUpdate"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID
    name: str
    organization: str
    description: str
    on_board: bool
    signup_enabled: bool
    admin_email: str


@serializable()
class NodeSettingsUpdate(PartialSyftObject):
    __canonical_name__ = "NodeSettingsUpdate"
    __version__ = SYFT_OBJECT_VERSION_3

    id: UID
    name: str
    organization: str
    description: str
    on_board: bool
    signup_enabled: bool
    admin_email: str
    association_request_auto_approval: bool


@serializable()
class NodeSettings(SyftObject):
    __canonical_name__ = "NodeSettings"
    __version__ = SYFT_OBJECT_VERSION_4
    __repr_attrs__ = [
        "name",
        "organization",
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
    description: str = "Text"
    node_type: NodeType = NodeType.DOMAIN
    signup_enabled: bool
    admin_email: str
    node_side_type: NodeSideType = NodeSideType.HIGH_SIDE
    show_warnings: bool
    association_request_auto_approval: bool
    default_worker_pool: str = DEFAULT_WORKER_POOL_NAME


@serializable()
class NodeSettingsV2(SyftObject):
    __canonical_name__ = "NodeSettings"
    __version__ = SYFT_OBJECT_VERSION_3
    __repr_attrs__ = [
        "name",
        "organization",
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
    description: str = "Text"
    node_type: NodeType = NodeType.DOMAIN
    signup_enabled: bool
    admin_email: str
    node_side_type: NodeSideType = NodeSideType.HIGH_SIDE
    show_warnings: bool


@migrate(NodeSettingsV2, NodeSettings)
def upgrade_node_settings() -> list[Callable]:
    return [make_set_default("association_request_auto_approval", False)]


@migrate(NodeSettings, NodeSettingsV2)
def downgrade_node_settings() -> list[Callable]:
    return [drop(["association_request_auto_approval"])]


@migrate(NodeSettingsUpdateV2, NodeSettingsUpdate)
def upgrade_node_settings_update() -> list[Callable]:
    return []


@migrate(NodeSettings, NodeSettingsV2)
def downgrade_node_settings_update() -> list[Callable]:
    return [drop(["association_request_auto_approval"])]
