# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject


class NodeConnection(SyftObject):
    __canonical_name__ = "NodeConnection"
    __version__ = SYFT_OBJECT_VERSION_1

    def get_cache_key() -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__}"
