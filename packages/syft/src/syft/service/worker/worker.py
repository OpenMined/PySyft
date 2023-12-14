# relative
import os
from ...serde.serializable import serializable
from ...store.document_store import SYFT_OBJECT_VERSION_1
from ...store.document_store import SyftObject
from ...types.datetime import DateTime


@serializable()
class DockerWorker(SyftObject):
    # version
    __canonical_name__ = "ContainerImage"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["container_id"]
    __attr_unique__ = ["container_id"]
    __repr_attrs__ = ["container_id", "created_at"]

    container_id: str
    created_at: DateTime = DateTime.now()

    def get_name(self):
        return os.getenv("DOCKER_WORKER_NAME", None)

    def _coll_repr_(self):
        return {
            "container_name": self.get_name(),
            "container_id": self.container_id,
            "created_at": self.created_at,
            "jobs": "TODO"
        }
