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
from ...client.api import APIRegistry
from ...serde.serializable import serializable
from ...store.linked_obj import LinkedObject
from ...types.base import SyftBaseModel
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.syft_object import short_uid
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.fonts import ITABLES_CSS
from ...util.fonts import fonts_css
from ..response import SyftError
from .worker_image import SyftWorkerImage


@serializable()
class WorkerStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    STOPPED = "Stopped"
    RESTARTED = "Restarted"


@serializable()
class ConsumerState(Enum):
    IDLE = "Idle"
    CONSUMING = "Consuming"


@serializable()
class WorkerHealth(Enum):
    HEALTHY = "✅"
    UNHEALTHY = "❌"


@serializable()
class SyftWorker(SyftObject):
    __canonical_name__ = "SyftWorker"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["name"]
    __attr_searchable__ = ["name", "container_id"]
    __repr_attrs__ = [
        "name",
        "container_id",
        "image",
        "status",
        "healthcheck",
        "worker_pool_name",
        "created_at",
    ]

    id: UID
    name: str
    container_id: Optional[str]
    created_at: DateTime = DateTime.now()
    healthcheck: Optional[WorkerHealth]
    status: WorkerStatus
    image: Optional[SyftWorkerImage]
    worker_pool_name: str
    consumer_state: ConsumerState = ConsumerState.IDLE
    job_id: Optional[UID]

    def get_job_repr(self):
        if self.job_id is not None:
            api = APIRegistry.api_for(
                node_uid=self.syft_node_location,
                user_verify_key=self.syft_client_verify_key,
            )
            job = api.services.job.get(self.job_id)
            if job.action.user_code_id is not None:
                func_name = api.services.code.get_by_id(
                    job.action.user_code_id
                ).service_func_name
                return f"{func_name} ({short_uid(self.job_id)})"
            else:
                return f"action ({short_uid(self.job_id)})"
        else:
            return ""

    def get_status_healthcheck(self) -> None:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )

        res = api.services.worker.status(worker_id=self.id)
        if isinstance(res, SyftError):
            return res
        self.status = res
        self.healthcheck = _get_healthcheck_based_on_status(status=self.status)

    def _coll_repr_(self) -> Dict[str, Any]:
        self.get_status_healthcheck()
        if self.image and self.image.image_identifier:
            image_name_with_tag = self.image.image_identifier.full_name_with_tag
        else:
            image_name_with_tag = "In Memory Worker"
        return {
            "Name": self.name,
            "Image": image_name_with_tag,
            "Healthcheck (health / unhealthy)": f"{self.healthcheck.value}",
            "Status": f"{self.status.value}",
            "Job": self.get_job_repr(),
            "Created at": str(self.created_at),
            "Container id": self.container_id,
            "Consumer state": str(self.consumer_state.value.lower()),
        }


@serializable()
class WorkerPool(SyftObject):
    __canonical_name__ = "WorkerPool"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["name"]
    __attr_searchable__ = ["name"]
    __repr_attrs__ = [
        "name",
        "image",
        "max_count",
        "workers",
        "created_at",
    ]

    name: str
    image: Optional[SyftWorkerImage]
    max_count: int
    worker_list: List[LinkedObject]
    created_at: DateTime = DateTime.now()

    @property
    def running_workers(self) -> Union[List[UID], SyftError]:
        """
        Called by the client. Query the running workers using an API call to the server
        """
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        running_workers = []
        for worker in self.worker_list:
            res = api.services.worker_pool.get_worker_status(
                worker_pool_id=self.id, worker_id=worker.object_uid
            )
            if isinstance(res, SyftError):
                return res
            if res == WorkerStatus.RUNNING:
                running_workers.append(worker.object_uid)

        return running_workers

    @property
    def healthy_workers(self) -> Union[List[UID], SyftError]:
        """
        Query the healthy workers using an API call to the server
        """
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )
        healthy_workers = []
        for worker in self.worker_list:
            res = api.services.worker_pool.get_worker_status(
                worker_pool_id=self.id, worker_id=worker.object_uid
            )
            if isinstance(res, SyftError):
                return res
            if res in (WorkerStatus.PENDING, WorkerStatus.RUNNING):
                healthy_workers.append(worker.object_uid)

        return healthy_workers

    def _coll_repr_(self) -> Dict[str, Any]:
        if self.image and self.image.image_identifier:
            return {
                "Pool Name": self.name,
                "Workers": len(self.workers),
                "Healthy (healthy / all)": f"{len(self.healthy_workers)} / {self.max_count}",
                "Running (running / all)": f"{len(self.running_workers)} / {self.max_count}",
                "Image": self.image.image_identifier.full_name_with_tag,
                "Created at": str(self.created_at),
            }
        else:
            return {
                "Pool Name": self.name,
                "Workers": len(self.workers),
                "Healthy (healthy / all)": f"{len(self.healthy_workers)} / {self.max_count}",
                "Running (running / all)": f"{len(self.running_workers)} / {self.max_count}",
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


def _get_worker_container(
    client: docker.DockerClient,
    worker: SyftWorker,
) -> Union[Container, SyftError]:
    try:
        return cast(Container, client.containers.get(worker.container_id))
    except docker.errors.NotFound as e:
        return SyftError(f"Worker {worker.id} container not found. Error {e}")
    except docker.errors.APIError as e:
        return SyftError(
            f"Unable to access worker {worker.id} container. "
            + f"Container server error {e}"
        )


_CONTAINER_STATUS_TO_WORKER_STATUS: Dict[str, WorkerStatus] = dict(
    [
        ("running", WorkerStatus.RUNNING),
        *(
            (status, WorkerStatus.STOPPED)
            for status in ["paused", "removing", "exited", "dead"]
        ),
        ("restarting", WorkerStatus.RESTARTED),
        ("created", WorkerStatus.PENDING),
    ]
)


def _get_worker_container_status(
    client: docker.DockerClient,
    worker: SyftWorker,
    container: Optional[Container] = None,
) -> Union[Container, SyftError]:
    if container is None:
        container = _get_worker_container(client, worker)

    if isinstance(container, SyftError):
        return container

    container_status = container.status

    return _CONTAINER_STATUS_TO_WORKER_STATUS.get(
        container_status,
        SyftError(message=f"Unknown container status: {container_status}"),
    )


def _get_healthcheck_based_on_status(status: WorkerStatus) -> WorkerHealth:
    if status in [WorkerStatus.PENDING, WorkerStatus.RUNNING]:
        return WorkerHealth.HEALTHY
    else:
        return WorkerHealth.UNHEALTHY
