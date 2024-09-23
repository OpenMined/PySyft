# stdlib
from collections.abc import Callable
from enum import Enum
from typing import Any
from typing import cast

# third party
import docker
from docker.models.containers import Container

# relative
from ...serde.serializable import serializable
from ...store.linked_obj import LinkedObject
from ...types.base import SyftBaseModel
from ...types.datetime import DateTime
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.syft_object import short_uid
from ...types.transforms import TransformContext
from ...types.uid import UID
from ..response import SyftError
from .worker_image import SyftWorkerImage
from .worker_image import SyftWorkerImageV1


@serializable(canonical_name="WorkerStatus", version=1)
class WorkerStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    STOPPED = "Stopped"
    RESTARTED = "Restarted"


@serializable(canonical_name="ConsumerState", version=1)
class ConsumerState(Enum):
    IDLE = "Idle"
    CONSUMING = "Consuming"
    DETACHED = "Detached"


@serializable(canonical_name="WorkerHealth", version=1)
class WorkerHealth(Enum):
    HEALTHY = "✅"
    UNHEALTHY = "❌"


@serializable()
class SyftWorkerV1(SyftObject):
    __canonical_name__ = "SyftWorker"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["name"]
    __attr_searchable__ = ["name", "container_id", "to_be_deleted"]
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
    container_id: str | None = None
    created_at: DateTime = DateTime.now()
    healthcheck: WorkerHealth | None = None
    status: WorkerStatus
    image: SyftWorkerImageV1 | None = None
    worker_pool_name: str
    consumer_state: ConsumerState = ConsumerState.DETACHED
    job_id: UID | None = None
    to_be_deleted: bool = False


@serializable()
class SyftWorker(SyftObject):
    __canonical_name__ = "SyftWorker"
    __version__ = SYFT_OBJECT_VERSION_2

    __attr_unique__ = ["name"]
    __attr_searchable__ = ["name", "container_id", "to_be_deleted"]
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
    container_id: str | None = None
    created_at: DateTime = DateTime.now()
    healthcheck: WorkerHealth | None = None
    status: WorkerStatus
    image: SyftWorkerImage | None = None
    worker_pool_name: str
    consumer_state: ConsumerState = ConsumerState.DETACHED
    job_id: UID | None = None
    to_be_deleted: bool = False

    @property
    def logs(self) -> str:
        return self.get_api().services.worker.logs(uid=self.id)

    def get_job_repr(self) -> str:
        if self.job_id is not None:
            api = self.get_api()
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

    def refresh_status(self) -> None:
        res = self.get_api().services.worker.status(uid=self.id)
        self.status, self.healthcheck = res
        return None

    def _coll_repr_(self) -> dict[str, Any]:
        self.refresh_status()

        if self.image and self.image.image_identifier:
            image_name_with_tag = self.image.image_identifier.full_name_with_tag
        else:
            image_name_with_tag = "In Memory Worker"

        healthcheck = self.healthcheck.value if self.healthcheck is not None else ""

        return {
            "Name": self.name,
            "Image": image_name_with_tag,
            "Healthcheck (health / unhealthy)": f"{healthcheck}",
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
    __attr_searchable__ = ["name", "image_id"]
    __repr_attrs__ = [
        "name",
        "image",
        "max_count",
        "workers",
        "created_at",
    ]
    __table_sort_attr__ = "Created at"

    name: str
    image_id: UID | None = None
    max_count: int
    worker_list: list[LinkedObject]
    created_at: DateTime = DateTime.now()

    @property
    def image(self) -> SyftWorkerImage | None:
        """
        Get the pool's image using the worker_image service API. This way we
        get the latest state of the image from the SyftWorkerImageStash
        """
        api = self.get_api_wrapped()
        if api.is_ok() and api.unwrap().services is not None:
            api = api.unwrap()
            return api.services.worker_image.get_by_uid(uid=self.image_id)
        else:
            return None

    @property
    def running_workers(self) -> list[SyftWorker]:
        """Query the running workers using an API call to the server"""
        _running_workers = [
            worker for worker in self.workers if worker.status == WorkerStatus.RUNNING
        ]

        return _running_workers

    @property
    def healthy_workers(self) -> list[SyftWorker]:
        """
        Query the healthy workers using an API call to the server
        """
        _healthy_workers = [
            worker
            for worker in self.workers
            if worker.healthcheck == WorkerHealth.HEALTHY
        ]

        return _healthy_workers

    def _coll_repr_(self) -> dict[str, Any]:
        if self.image and self.image.image_identifier:
            image_name_with_tag = self.image.image_identifier.full_name_with_tag
        else:
            image_name_with_tag = "In Memory Worker"
        return {
            "Pool Name": self.name,
            "Workers": len(self.workers),
            "Healthy (healthy / all)": f"{len(self.healthy_workers)} / {self.max_count}",
            "Running (running / all)": f"{len(self.running_workers)} / {self.max_count}",
            "Image": image_name_with_tag,
            "Created at": str(self.created_at),
        }

    def _repr_html_(self) -> str:
        return f"""
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
    def workers(self) -> list[SyftWorker]:
        resolved_workers = []
        for worker in self.worker_list:
            try:
                resolved_worker = worker.resolve
            except SyftException:
                resolved_worker = None
            if resolved_worker is None:
                continue
            resolved_worker.refresh_status()
            resolved_workers.append(resolved_worker)
        return resolved_workers


@serializable(canonical_name="WorkerOrchestrationType", version=1)
class WorkerOrchestrationType(Enum):
    DOCKER = "docker"
    KUBERNETES = "k8s"
    PYTHON = "python"


@serializable(canonical_name="ContainerSpawnStatus", version=1)
class ContainerSpawnStatus(SyftBaseModel):
    __repr_attrs__ = ["worker_name", "worker", "error"]

    worker_name: str
    worker: SyftWorker | None = None
    error: str | None = None


@as_result(SyftException)
def _get_worker_container(
    client: docker.DockerClient,
    worker: SyftWorker,
) -> Container:
    try:
        return cast(Container, client.containers.get(worker.container_id))
    except docker.errors.NotFound as e:
        raise SyftException(
            public_message=f"Worker {worker.id} container not found. Error {e}"
        )
    except docker.errors.APIError as e:
        raise SyftException(
            public_message=f"Unable to access worker {worker.id} container. "
            + f"Container server error {e}"
        )


_CONTAINER_STATUS_TO_WORKER_STATUS: dict[str, WorkerStatus] = dict(
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


@as_result(SyftException)
def _get_worker_container_status(
    client: docker.DockerClient,
    worker: SyftWorker,
    container: Container | None = None,
) -> Container:
    if container is None:
        container = _get_worker_container(client, worker).unwrap()
    container_status = container.status

    return _CONTAINER_STATUS_TO_WORKER_STATUS.get(
        container_status,
        SyftError(message=f"Unknown container status: {container_status}"),
    )


def migrate_worker_image_v1_to_v2(context: TransformContext) -> TransformContext:
    old_image = context["image"]
    if isinstance(old_image, SyftWorkerImageV1):
        new_image = old_image.migrate_to(
            version=SYFT_OBJECT_VERSION_2,
            context=context.to_server_context(),
        )
        context["image"] = new_image
    return context


@migrate(SyftWorkerV1, SyftWorker)
def migrate_worker_v1_to_v2() -> list[Callable]:
    return [migrate_worker_image_v1_to_v2]
