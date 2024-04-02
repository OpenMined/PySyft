# stdlib

# relative
from ...abstract_node import NodeSideType
from ...abstract_node import NodeType
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import SyftObject
from ...types.uid import UID


@serializable()
class NodeSettingsUpdate(PartialSyftObject):
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
