# stdlib
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import cast

# third party
import docker
from docker.models.containers import Container

# relative
from ...serde.serializable import serializable
from ...types.base import SyftBaseModel
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.fonts import ITABLES_CSS
from ...util.fonts import fonts_css
from ..response import SyftError


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
        "full_image_tag_str",
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
    full_image_tag_str: Optional[str]

    def get_status_healthcheck(self) -> None:
        self.status: WorkerStatus = get_worker_container_status(self)
        self.healthcheck = get_healthcheck_based_on_status(status=self.status)

    def _coll_repr_(self) -> Dict[str, Any]:
        self.get_status_healthcheck()
        return {
            "Name": self.name,
            "Image": self.full_image_tag_str,
            "Healthcheck (health / unhealthy)": f"{self.healthcheck.value}",
            "Status": f"{self.status.value}",
            "Created at": str(self.created_at),
        }


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
    syft_worker_image_name_tag: Optional[str]
    max_count: int
    workers: List[SyftWorker]
    created_at: DateTime = DateTime.now()

    @property
    def running_workers(self) -> List[SyftWorker]:
        return [
            worker
            for worker in self.workers
            if get_worker_container_status(worker) == WorkerStatus.RUNNING
        ]

    @property
    def healthy_workers(self) -> List[SyftWorker]:
        return [
            worker
            for worker in self.workers
            if (
                get_worker_container_status(worker) == WorkerStatus.PENDING
                or get_worker_container_status(worker) == WorkerStatus.RUNNING
            )
        ]

    def _coll_repr_(self) -> Dict[str, Any]:
        return {
            "Pool Name": self.name,
            "Workers": len(self.workers),
            "Healthy (healthy / all)": f"{len(self.healthy_workers)} / {self.max_count}",
            "Running (running / all)": f"{len(self.running_workers)} / {self.max_count}",
            "Image": self.syft_worker_image_name_tag,
            "Created at": str(self.created_at),
        }

    def _repr_html_(self) -> Any:
        return f"""
            <style>
            {fonts_css}
            .syft-dataset {{color: {SURFACE[options.color_theme]};}}
            .syft-dataset h3,
            .syft-dataset p
              {{font-family: 'Open Sans';}}
              {ITABLES_CSS}
            </style>
            <div class='syft-dataset'>
            <h3>{self.name}</h3>
            <p class='paragraph-sm'>
                <strong><span class='pr-8'>Created on: </span></strong>
                {self.created_at}
            </p>
            <p class='paragraph-sm'>
                <strong><span class='pr-8'>Healthy Workers:</span></strong>
                {len(self.healthy_workers)} / {self.max_count}
            </p>
            <p class='paragraph-sm'>
                <strong><span class='pr-8'>Running Workers:</span></strong>
                {len(self.running_workers)} / {self.max_count}
            </p>
            {self.workers._repr_html_()}
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


def get_worker_container(
    worker: SyftWorker, docker_client: Optional[docker.DockerClient] = None
) -> Union[Container, SyftError]:
    docker_client = docker_client if docker_client is not None else docker.from_env()
    try:
        return cast(Container, docker_client.containers.get(worker.container_id))
    except docker.errors.NotFound as e:
        return SyftError(f"Worker {worker.id} container not found. Error {e}")
    except docker.errors.APIError as e:
        return SyftError(
            f"Unable to access worker {worker.id} container. "
            + f"Container server error {e}"
        )


def get_worker_container_status(
    worker: SyftWorker, docker_client: Optional[docker.DockerClient] = None
) -> Union[WorkerStatus, SyftError]:
    container = get_worker_container(worker, docker_client)
    if isinstance(container, SyftError):
        return container

    container_status = container.status
    syft_container_status = None
    if container_status == "running":
        syft_container_status = WorkerStatus.RUNNING
    elif container_status in ["paused", "removing", "exited", "dead"]:
        syft_container_status = WorkerStatus.STOPPED
    elif container_status == "restarting":
        syft_container_status = WorkerStatus.RESTARTED
    elif container_status == "created":
        syft_container_status = WorkerStatus.PENDING
    else:
        return SyftError(message=f"Unknown container status: {container_status}")

    return syft_container_status


def get_healthcheck_based_on_status(status: WorkerStatus) -> WorkerHealth:
    if status in [WorkerStatus.PENDING, WorkerStatus.RUNNING]:
        return WorkerHealth.HEALTHY
    else:
        return WorkerHealth.UNHEALTHY
