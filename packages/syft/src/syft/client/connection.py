# stdlib
from typing import Any

# relative
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..types.uid import UID


class ServerConnection(SyftObject):
    __canonical_name__ = "ServerConnection"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID | None = None  # type: ignore

    def get_cache_key(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__}"

    @property
    def route(self) -> Any:
        # relative
        from ..service.network.routes import connection_to_route

        return connection_to_route(self)
