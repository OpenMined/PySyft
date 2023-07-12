# stdlib

# relative
from ...serde.serializable import serializable
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID


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


@serializable()
class NodeSettings(SyftObject):
    __canonical_name__ = "NodeSettings"
    __version__ = SYFT_OBJECT_VERSION_1
    __repr_attrs__ = ["name", "organization", "deployed_on", "signup_enabled"]

    name: str = "Node"
    deployed_on: str
    organization: str = "OpenMined"
    on_board: bool = True
    description: str = "Text"
    signup_enabled: bool
