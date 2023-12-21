# stdlib
from enum import Enum
from typing import List
from typing import Optional

# relative
from ...serde.serializable import serializable
from ...store.linked_obj import LinkedObject
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
    __attr_searchable__ = ["name", "container_id", "image_hash", "worker_pool_name"]

    id: UID
    name: str
    container_id: Optional[str]
    created_at: DateTime = DateTime.now()
    image_hash: Optional[str]
    healthcheck: Optional[WorkerHealth]
    status: WorkerStatus
    worker_pool_name: str


@serializable()
class WorkerPool(SyftObject):
    __canonical_name__ = "WorkerPool"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["name"]
    __attr_searchable__ = ["name", "syft_worker_image_id"]

    name: str
    syft_worker_image_id: UID
    max_count: int
    worker_list: List[LinkedObject]

    @property
    def workers(self):
        return [worker.resolve for worker in self.worker_list]


@serializable()
class WorkerOrchestrationType:
    DOCKER = "docker"
    K8s = "k8s"
    PYTHON = "python"


@serializable()
class ContainerSpawnStatus(SyftBaseModel):
    __repr_attrs__ = ["worker_name", "worker", "error"]

    worker_name: str
    worker: Optional[SyftWorker]
    error: Optional[str]
