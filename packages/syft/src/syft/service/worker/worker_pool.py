# stdlib
from enum import Enum
from typing import List
from typing import Optional

# relative
from ...serde.serializable import serializable
from ...types.base import SyftBaseModel
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID


@serializable()
class WorkerStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    STOPPED = "Stopped"
    RESTARTED = "Restarted"


@serializable()
class WorkerHealth(Enum):
    HEALTHY = "✅"
    UNHEALTHY = "❌"


@serializable()
class SyftWorker(SyftObject):
    __canonical_name__ = "SyftWorker"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["name"]
    __attr_searchable__ = ["name", "container_id", "image_hash"]
    __repr_attrs__ = [
        "name",
        "container_id",
        "image_hash",
        "status",
        "healthcheck",
        "created_at",
    ]

    id: UID
    name: str
    container_id: str
    created_at: DateTime = DateTime.now()
    image_hash: str
    healthcheck: Optional[WorkerHealth]
    status: WorkerStatus


@serializable()
class WorkerPool(SyftObject):
    __canonical_name__ = "WorkerPool"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["name"]
    __attr_searchable__ = ["name", "syft_worker_image_id"]
    __repr_attrs__ = [
        "name",
        "syft_worker_image_id",
        "max_count",
        "workers",
        "created_at",
    ]

    name: str
    syft_worker_image_id: UID
    max_count: int
    workers: List[SyftWorker]
    created_at: DateTime = DateTime.now()


@serializable()
class WorkerOrchestrationType:
    DOCKER = "docker"
    K8s = "k8s"


@serializable()
class ContainerSpawnStatus(SyftBaseModel):
    __repr_attrs__ = ["worker_name", "worker", "error"]

    worker_name: str
    worker: Optional[SyftWorker]
    error: Optional[str]
