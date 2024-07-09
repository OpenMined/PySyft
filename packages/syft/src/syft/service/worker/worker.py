# stdlib
from typing import Any

# relative
from ...serde.serializable import serializable
from ...store.document_store import SYFT_OBJECT_VERSION_1
from ...store.document_store import SyftObject
from ...types.datetime import DateTime


@serializable()
class DockerWorker(SyftObject):
    # version
    __canonical_name__ = "ContainerImage"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["container_id", "container_name"]
    __attr_unique__ = ["container_id"]
    __repr_attrs__ = ["container_id", "created_at"]

    container_name: str
    container_id: str
    created_at: DateTime = DateTime.now()

    def _coll_repr_(self) -> dict[str, Any]:
        return {
            "container_name": self.container_name,
            "container_id": self.container_id,
            "created_at": self.created_at,
        }
