# stdlib
from typing import List
from typing import Union

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from .utils import run_containers
from .worker_image_stash import SyftWorkerImageStash
from .worker_pool import ContainerSpawnStatus
from .worker_pool import WorkerOrchestrationType
from .worker_pool import WorkerPool
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
