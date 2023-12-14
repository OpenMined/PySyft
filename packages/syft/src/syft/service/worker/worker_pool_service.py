# stdlib
from typing import List
from typing import Union

# third party
import docker

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
        worker_pool = self.stash.get_by_uid(
            credentials=context.credentials, uid=worker_pool_id
        )
        if worker_pool.is_err():
            return SyftError(message=f"{worker_pool.err()}")

        worker_pool: WorkerPool = worker_pool.ok()
        worker = None
        for w in worker_pool.workers:
            if w.id == worker_id:
                worker = w
                break
        if worker is None:
            return SyftError(
                message=f"Worker with id: {worker_id} not found in pool: {worker_pool.name}"
            )

        # delete the worker using docker client sdk
        docker_client = docker.from_env()
        docker_container = docker_client.containers.get(worker.container_id)
        try:
            # stop the container
            docker_container.stop()
            # Remove the container and its volumes
            docker_container.remove(force=force, v=True)
        except docker.errors.APIError as e:
            if "removal of container" in str(e) and "is already in progress" in str(e):
                # If the container is already being removed, ignore the error
                pass
            else:
                # If it's a different error, return it
                return SyftError(
                    message=f"Failed to delete worker with id: {worker_id}. Error: {e}"
                )
        except Exception as e:
            return SyftError(
                message=f"Failed to delete worker with id: {worker_id}. Error: {e}"
            )

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
        self, context: AuthedServiceContext, pool_name: str, worker_id: UID
    ) -> Union[SyftWorker, SyftError]:
        result = self.stash.get_by_name(context.credentials, pool_name=pool_name)
        if result.is_err():
            return SyftError(message=f"{result.err()}")
        if result.ok() is None:
            return SyftError(
                message=f"Worker Pool with name: {pool_name} does not exist !!"
            )

        worker_pool = result.ok()

        found_worker = None
        for worker in worker_pool.workers:
            if worker.id == worker_id:
                found_worker = worker
                break

        if found_worker is None:
            return SyftError(message=f"Worker with id: {worker_id} does not exist !!")

        worker_status = self.get_worker_status(
            context=context, pool_name=pool_name, worker_id=found_worker.id
        )
        if isinstance(worker_status, SyftError):
            return worker_status
        elif worker_status != WorkerStatus.PENDING:
            for worker in worker_pool.workers:
                if worker.id == worker_id:
                    worker.status = worker_status
                    break
            result = self.stash.update(
                credentials=context.credentials,
                obj=worker_pool,
            )
            if result.is_err():
                return SyftError(message=f"{result.err()}")
            else:
                return found_worker
        else:
            return found_worker

    @service_method(
        path="worker_pool.get_worker_status",
        name="get_worker_status",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_worker_status(
        self, context: AuthedServiceContext, pool_name: str, worker_id: UID
    ) -> Union[WorkerStatus, SyftError]:
        result = self.stash.get_by_name(context.credentials, pool_name=pool_name)
        if result.is_err():
            return SyftError(message=f"{result.err()}")

        if result.ok() is None:
            return SyftError(
                message=f"Worker Pool with name: {pool_name} does not exist !!"
            )

        worker_pool = result.ok()

        found_worker = None
        for worker in worker_pool.workers:
            if worker.id == worker_id:
                found_worker = worker
                break

        if found_worker is None:
            return SyftError(message=f"Worker with id: {worker_id} does not exist !!")

        client = docker.from_env()
        worker_status = client.containers.get(found_worker.container_id).status

        if worker_status == "running":
            found_worker.status = WorkerStatus.RUNNING
        elif worker_status in ["paused", "removing", "exited", "dead"]:
            found_worker.status = WorkerStatus.STOPPED
        elif worker_status["State"]["Status"] == "restarting":
            found_worker.status = WorkerStatus.RESTARTED
        elif worker_status["State"]["Status"] == "created":
            found_worker.status = WorkerStatus.PENDING

        client.close()

        return found_worker.status
