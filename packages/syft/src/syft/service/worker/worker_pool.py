# stdlib
from enum import Enum
from typing import List
from typing import Optional

# relative
from ...serde.serializable import serializable
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftBaseObject
from ...types.syft_object import SyftObject
from ...types.uid import UID


class WorkerStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    STOPPED = "Stopped"
    RESTARTED = "Restarted"


class WorkerHealth(Enum):
    HEALTHY = "✅"
    UNHEALTHY = "❌"


class SyftWorker(SyftObject):
    __canonical_name__ = "SyftWorker"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    container_id: str
    created_at: DateTime = DateTime.now()
    image_hash: str
    healthcheck: WorkerHealth
    status: WorkerStatus


class WorkerPool(SyftObject):
    __canonical_name__ = "WorkerPool"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    syft_worker_image_id: UID
    max_count: int
    workers: List[SyftWorker]


class WorkerOrchestrationType:
    DOCKER = "docker"
    K8s = "k8s"


@serializable()
class ContainerSpawnStatus(SyftBaseObject):
    worker: SyftWorker
    error: Optional[str]
