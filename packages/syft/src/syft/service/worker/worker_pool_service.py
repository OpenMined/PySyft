# stdlib
from typing import Union

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
    ) -> Union[SyftSuccess, SyftError]:
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

        Returns:
            Union[SyftSuccess, SyftError]: Returns success message if all workers are
            created otherwise returns the corresponding error message.
        """

        result = self.image_stash.get_by_uid(context=context.credentials, uid=image_uid)
        if result.is_err():
            return SyftError(
                message=f"Failed to retrieve Worker Image with id: {image_uid}. Error: {result.err()}"
            )

        worker_image = result.ok()

        workers = run_containers(
            pool_name=name,
            worker_image=worker_image,
            number=number,
        )

        workers_spinned = []

        for worker in workers:
            if isinstance(worker, SyftError):
                continue
            workers_spinned.append(worker)

        worker_pool = WorkerPool(
            name=name,
            syft_worker_image_id=image_uid,
            max_count=len(workers_spinned),
            workers=workers_spinned,
        )
        result = self.stash.set(context.credentials, worker_pool)

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        return SyftSuccess(message="Worker pool successfully created.")
