# stdlib
from collections.abc import Callable
from typing import Any

# relative
from ...serde.serializable import serializable
from ...store.document_store import SYFT_OBJECT_VERSION_2
from ...store.document_store import SyftObject
from ...types.datetime import DateTime
from ...types.syft_migration import migrate
from ...types.transforms import drop
from ...types.transforms import make_set_default


@serializable()
class DockerWorkerV1(SyftObject):
    # version
    __canonical_name__ = "ContainerImage"
    __version__ = SYFT_OBJECT_VERSION_2

    __attr_searchable__ = ["container_id"]
    __attr_unique__ = ["container_id"]
    __repr_attrs__ = ["container_id", "created_at"]

    container_id: str
    created_at: DateTime = DateTime.now()


@serializable()
class DockerWorker(SyftObject):
    # version
    __canonical_name__ = "ContainerImage"
    __version__ = SYFT_OBJECT_VERSION_2

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


@migrate(DockerWorker, DockerWorkerV1)
def downgrade_job_v2_to_v1() -> list[Callable]:
    return [drop(["container_name"])]


@migrate(DockerWorkerV1, DockerWorker)
def upgrade_job_v2_to_v3() -> list[Callable]:
    return [make_set_default("job_consumer_id", None)]
