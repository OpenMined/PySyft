from typing import Optional

from ...serde.serializable import serializable
from ...types.syft_object import SyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1

@serializable()
class NodeSettingsUpdate(SyftObject):
    __canonical_name__ = "NodeSettingsUpdate"
    __version__ = SYFT_OBJECT_VERSION_1

    organization: Optional[str]
    description: Optional[str]
    on_board: Optional[bool]


@serializable()
class NodeSettings(SyftObject):
    __canonical_name__ = "NodeSettings"
    __version__ = SYFT_OBJECT_VERSION_1

    deployed_on: str
    organization: str = "OpenMined"
    on_board: bool = False
    description: str = "Text"
