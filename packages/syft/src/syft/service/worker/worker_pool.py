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
from ...util import options
from ...util.colors import SURFACE


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

    def _repr_html_(self) -> str:
        return f"""
            <style>
            .syft-worker {{color: {SURFACE[options.color_theme]};}}
            </style>
            <div class='syft-worker' style='line-height:25%'>
                <h3>SyftWorker</h3>
                <p><strong>ID: </strong>{self.id}</p>
                <p><strong>Name: </strong>{self.name}</p>
                <p><strong>Container ID: </strong>{self.container_id}</p>
                <p><strong>Image Hash: </strong>{self.image_hash}</p>
                <p><strong>Healthcheck: </strong>{self.healthcheck}</p>
                <p><strong>Status: </strong>{self.status}</p>
                <p><strong>Created At: </strong>{self.created_at}</p>
            </div>
            """


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

    def _repr_html_(self) -> str:
        return f"""
            <style>
            .syft-worker-pool {{color: {SURFACE[options.color_theme]};}}
            </style>
            <div class='syft-worker-pool' style='line-height:25%'>
                <h3>SyftWorkerPool</h3>
                <p><strong>Name: </strong>{self.name}</p>
                <p><strong>Syft worker image id: </strong>{self.syft_worker_image_id}</p>
                <p><strong>Max Count: </strong>{str(self.max_count)}</p>
                <p><strong>Workers: </strong>{self.workers}</p>
                <p><strong>Created At: </strong>{self.created_at}</p>
            </div>
            """


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
