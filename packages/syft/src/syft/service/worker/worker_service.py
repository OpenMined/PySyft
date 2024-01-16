# stdlib
import contextlib
from typing import List
from typing import Tuple
from typing import Union
from typing import cast

# third party
import docker

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.document_store import SyftSuccess
from ...types.uid import UID
from ...util.telemetry import instrument
from ..service import AbstractService
from ..service import AuthedServiceContext
from ..service import SyftError
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .utils import DEFAULT_WORKER_POOL_NAME
from .utils import _get_healthcheck_based_on_status
from .worker_pool import ContainerSpawnStatus
from .worker_pool import SyftWorker
from .worker_pool import WorkerHealth
from .worker_pool import WorkerStatus
from .worker_pool import _get_worker_container_status
from .worker_stash import WorkerStash


@instrument
@serializable()
class WorkerService(AbstractService):
    store: DocumentStore
    stash: WorkerStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = WorkerStash(store=store)

    @service_method(
        path="worker.start_workers",
        name="start_workers",
        roles=ADMIN_ROLE_LEVEL,
    )
    def start_workers(
        self, context: AuthedServiceContext, n: int = 1
    ) -> Union[List[ContainerSpawnStatus], SyftError]:
        """Add a Container Image."""
        worker_pool_service = context.node.get_service("SyftWorkerPoolService")

        return worker_pool_service.add_workers(
            context, number=n, pool_name=DEFAULT_WORKER_POOL_NAME
        )

    @service_method(
        path="worker.get_all", name="get_all", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def list(self, context: AuthedServiceContext) -> Union[SyftSuccess, SyftError]:
        """List all the workers."""
        result = self.stash.get_all(context.credentials)

        if result.is_err():
            return SyftError(message=f"Failed to fetch workers. {result.err()}")

        workers = result.ok()

        if context.node.in_memory_workers:
            return workers

        # If container workers, check their statuses
        with contextlib.closing(docker.from_env()) as client:
            for idx, worker in enumerate(workers):
                worker_ = _check_and_update_status_for_worker(
                    client=client,
                    worker=worker,
                    worker_stash=self.stash,
                    credentials=context.credentials,
                )

                if not isinstance(worker_, SyftWorker):
                    return worker_

                workers[idx] = worker_

        return workers

    @service_method(
        path="worker.status", name="status", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def status(
        self,
        context: AuthedServiceContext,
        uid: UID,
    ) -> Union[Tuple[WorkerStatus, WorkerHealth], SyftError]:
        worker = self.get(context=context, uid=uid)

        if not isinstance(worker, SyftWorker):
            return worker

        return worker.status, worker.healthcheck

    @service_method(
        path="worker.get",
        name="get",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftWorker, SyftError]:
        result = self.stash.get_by_uid(credentials=context.credentials, uid=uid)
        if result.is_err():
            return SyftError(message=f"Failed to retrieve worker with UID {uid}")

        if result.ok() is None:
            return SyftError(message=f"Worker doesn't exists for uid: {uid}")

        worker = cast(SyftWorker, result.ok())

        if context.node.in_memory_workers:
            return worker

        with contextlib.closing(docker.from_env()) as client:
            return _check_and_update_status_for_worker(
                client=client,
                worker=worker,
                worker_stash=self.stash,
                credentials=context.credentials,
            )


def _check_and_update_status_for_worker(
    client: docker.DockerClient,
    worker: SyftWorker,
    worker_stash: WorkerStash,
    credentials: SyftVerifyKey,
) -> Union[SyftWorker, SyftError]:
    worker_status = _get_worker_container_status(client, worker)

    if isinstance(worker_status, SyftError):
        return worker_status

    worker.status = worker_status

    worker.healthcheck = _get_healthcheck_based_on_status(status=worker_status)

    result = worker_stash.update(
        credentials=credentials,
        obj=worker,
    )

    return (
        SyftError(
            message=f"Failed to update status for worker: {worker.id}. Error: {result.err()}"
        )
        if result.is_err()
        else cast(SyftWorker, result.ok())
    )
