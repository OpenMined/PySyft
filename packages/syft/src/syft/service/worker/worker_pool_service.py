# stdlib
import contextlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

# third party
import docker
from docker.models.containers import Container

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from .utils import run_containers
from .worker_image_stash import SyftWorkerImageStash
from .worker_pool import ContainerSpawnStatus
from .worker_pool import SyftWorker
from .worker_pool import WorkerOrchestrationType
from .worker_pool import WorkerPool
from .worker_pool import WorkerStatus
from .worker_pool_stash import SyftWorkerPoolStash


@serializable()
class SyftWorkerPoolService(AbstractService):
    store: DocumentStore
    stash: SyftWorkerPoolStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = SyftWorkerPoolStash(store=store)
        self.image_stash = SyftWorkerImageStash(store=store)

    @service_method(
        path="worker_pool.create",
        name="create",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def create_pool(
        self,
        context: AuthedServiceContext,
        name: str,
        image_uid: UID,
        number: int,
        username: str = None,
        password: str = None
    ) -> Union[List[ContainerSpawnStatus], SyftError]:
        """Creates a pool of workers from the given SyftWorkerImage.

        - Retrieves the image for the given UID
        - Use docker to launch containers for given image
        - For each successful container instantiation create a SyftWorker object
        - Creates a SyftWorkerPool object

        Args:
            context (AuthedServiceContext): context passed to the service
            name (str): name of the pool
            image_id (UID): UID of the SyftWorkerImage against which the pool should be created
            number (int): number of SyftWorker that needs to be created in the pool
        """

        result = self.stash.get_by_name(context.credentials, pool_name=name)

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        if result.ok() is not None:
            return SyftError(message=f"Worker Pool with name: {name} already exists !!")

        result = self.image_stash.get_by_uid(
            credentials=context.credentials, uid=image_uid
        )
        if result.is_err():
            return SyftError(
                message=f"Failed to retrieve Worker Image with id: {image_uid}. Error: {result.err()}"
            )

        worker_image = result.ok()

        container_statuses: List[ContainerSpawnStatus] = run_containers(
            pool_name=name,
            worker_image=worker_image,
            number=number,
            orchestration=WorkerOrchestrationType.DOCKER,
            username=username,
            password=password,
        )

        workers = [
            container_status.worker
            for container_status in container_statuses
            if container_status.worker
        ]

        worker_pool = WorkerPool(
            name=name,
            syft_worker_image_id=image_uid,
            max_count=number,
            workers=workers,
        )
        result = self.stash.set(credentials=context.credentials, obj=worker_pool)

        if result.is_err():
            return SyftError(message=f"Failed to save Worker Pool: {result.err()}")

        return container_statuses

    @service_method(
        path="worker_pool.get_all",
        name="get_all",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[List[WorkerPool], SyftError]:
        # TODO: During get_all, we should dynamically make a call to docker to get the status of the containers
        # and update the status of the workers in the pool.
        result = self.stash.get_all(credentials=context.credentials)
        if result.is_err():
            return SyftError(message=f"{result.err()}")

        return result.ok()

    @service_method(
        path="worker_pool.delete_worker",
        name="delete_worker",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def delete_worker(
        self,
        context: AuthedServiceContext,
        worker_pool_id: UID,
        worker_id: UID,
        force: bool = False,
    ) -> Union[SyftSuccess, SyftError]:
        worker_pool_worker = self._get_worker_pool_and_worker(
            context, worker_pool_id, worker_id
        )
        if isinstance(worker_pool_worker, SyftError):
            return worker_pool_worker

        worker_pool, worker = worker_pool_worker

        # delete the worker using docker client sdk
        with contextlib.closing(docker.from_env()) as client:
            docker_container = _get_worker_container(client, worker)
            if isinstance(docker_container, SyftError):
                return docker_container

            stopped = _stop_worker_container(worker, docker_container, force)
            if stopped is not None:
                return stopped

        # remove the worker from the pool
        worker_pool.workers.remove(worker)
        result = self.stash.update(context.credentials, obj=worker_pool)
        if result.is_err():
            return SyftError(message=f"Failed to update worker pool: {result.err()}")

        return SyftSuccess(
            message=f"Worker with id: {worker_id} deleted successfully from pool: {worker_pool.name}"
        )

    @service_method(
        path="worker_pool.get_worker",
        name="get_worker",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_worker(
        self, context: AuthedServiceContext, worker_pool_id: UID, worker_id: UID
    ) -> Union[SyftWorker, SyftError]:
        worker_pool_worker = self._get_worker_pool_and_worker(
            context, worker_pool_id, worker_id
        )
        if isinstance(worker_pool_worker, SyftError):
            return worker_pool_worker

        worker_pool, worker = worker_pool_worker

        with contextlib.closing(docker.from_env()) as client:
            container = _get_worker_container(client, worker)
            if isinstance(container, SyftError):
                return container

            container_status = container.status

        worker_status = _CONTAINER_STATUS_TO_WORKER_STATUS.get(container_status)
        if worker_status is None:
            return SyftError(message=f"Unknown container status: {container_status}")

        if isinstance(worker_status, SyftError):
            return worker_status

        if worker_status != WorkerStatus.PENDING:
            worker.status = worker_status

            result = self.stash.update(
                credentials=context.credentials,
                obj=worker_pool,
            )

            return (
                SyftError(
                    message=f"Failed to update worker status. Error: {result.err()}"
                )
                if result.is_err()
                else worker
            )

        return worker

    @service_method(
        path="worker_pool.get_worker_status",
        name="get_worker_status",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_worker_status(
        self, context: AuthedServiceContext, worker_pool_id: UID, worker_id: UID
    ) -> Union[WorkerStatus, SyftError]:
        worker = self.get_worker(context, worker_pool_id, worker_id)
        return worker if isinstance(worker, SyftError) else worker.status

    @service_method(
        path="worker_pool.worker_logs",
        name="worker_logs",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def worker_logs(
        self,
        context: AuthedServiceContext,
        worker_pool_id: UID,
        worker_id: UID,
        raw: bool = False,
    ) -> Union[bytes, str, SyftError]:
        worker_pool_worker = self._get_worker_pool_and_worker(
            context, worker_pool_id, worker_id
        )
        if isinstance(worker_pool_worker, SyftError):
            return worker_pool_worker

        _, worker = worker_pool_worker

        with contextlib.closing(docker.from_env()) as client:
            docker_container = _get_worker_container(client, worker)
            if isinstance(docker_container, SyftError):
                return docker_container

            try:
                logs = cast(bytes, docker_container.logs())
            except docker.errors.APIError as e:
                return SyftError(
                    f"Failed to get worker {worker.id} container logs. Error {e}"
                )

        return logs if raw else logs.decode(errors="ignore")

    def _get_worker_pool(
        self,
        context: AuthedServiceContext,
        worker_pool_id: UID,
    ) -> Union[WorkerPool, SyftError]:
        worker_pool = self.stash.get_by_uid(
            credentials=context.credentials, uid=worker_pool_id
        )

        return (
            SyftError(message=f"{worker_pool.err()}")
            if worker_pool.is_err()
            else cast(WorkerPool, worker_pool.ok())
        )

    def _get_worker_pool_and_worker(
        self, context: AuthedServiceContext, worker_pool_id: UID, worker_id: UID
    ) -> Union[Tuple[WorkerPool, SyftWorker], SyftError]:
        worker_pool = self._get_worker_pool(context, worker_pool_id)
        if isinstance(worker_pool, SyftError):
            return worker_pool

        worker = _get_worker(worker_pool, worker_id)
        if isinstance(worker, SyftError):
            return worker

        return worker_pool, worker


def _get_worker_opt(worker_pool: WorkerPool, worker_id: UID) -> Optional[SyftWorker]:
    try:
        return next(worker for worker in worker_pool.workers if worker.id == worker_id)
    except StopIteration:
        return None


def _get_worker(
    worker_pool: WorkerPool, worker_id: UID
) -> Union[SyftWorker, SyftError]:
    worker = _get_worker_opt(worker_pool, worker_id)
    return (
        worker
        if worker is not None
        else SyftError(
            message=f"Worker with id: {worker_id} not found in pool: {worker_pool.name}"
        )
    )


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


def _stop_worker_container(
    worker: SyftWorker,
    container: Container,
    force: bool,
) -> Optional[SyftError]:
    try:
        # stop the container
        container.stop()
        # Remove the container and its volumes
        _remove_worker_container(container, force=force, v=True)
    except Exception as e:
        return SyftError(
            message=f"Failed to delete worker with id: {worker.id}. Error: {e}"
        )


def _remove_worker_container(container: Container, **kwargs: Any) -> None:
    try:
        container.remove(**kwargs)
    except docker.errors.NotFound:
        return
    except docker.errors.APIError as e:
        if "removal of container" in str(e) and "is already in progress" in str(e):
            # If the container is already being removed, ignore the error
            return
