# stdlib
import contextlib
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

# third party
import docker
from docker.models.containers import Container

# relative
from ...custom_worker.k8s import IN_KUBERNETES
from ...custom_worker.runner_k8s import KubernetesRunner
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
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .utils import DEFAULT_WORKER_POOL_NAME
from .utils import _get_healthcheck_based_on_status
from .utils import map_pod_to_worker_status
from .worker_pool import ContainerSpawnStatus
from .worker_pool import SyftWorker
from .worker_pool import WorkerHealth
from .worker_pool import WorkerStatus
from .worker_pool import _get_worker_container
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

        workers: List[SyftWorker] = result.ok()

        if context.node.in_memory_workers:
            return workers
        else:
            # If container workers, check their statuses
            workers = refresh_worker_status(workers, self.stash, context.credentials)

        return workers

    @service_method(
        path="worker.status", name="status", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def status(
        self,
        context: AuthedServiceContext,
        uid: UID,
    ) -> Union[Tuple[WorkerStatus, WorkerHealth], SyftError]:
        result = self.get(context=context, uid=uid)

        if isinstance(result, SyftError):
            return result

        return result.status, result.healthcheck

    @service_method(
        path="worker.get",
        name="get",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftWorker, SyftError]:
        worker = self._get_worker(context=context, uid=uid)
        if isinstance(worker, SyftError):
            return worker

        if context.node.in_memory_workers:
            return worker
        else:
            return refresh_worker_status([worker], self.stash, context.credentials)[0]

    @service_method(
        path="worker.logs",
        name="logs",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def logs(
        self,
        context: AuthedServiceContext,
        uid: UID,
        raw: bool = False,
    ) -> Union[bytes, str, SyftError]:
        worker = self._get_worker(context=context, uid=uid)
        if isinstance(worker, SyftError):
            return worker

        if context.node.in_memory_workers:
            logs = b"Logs not implemented for In Memory Workers"
        elif IN_KUBERNETES:
            runner = KubernetesRunner()
            return runner.get_pod_logs(pod_name=worker.name)
        else:
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

    @service_method(
        path="worker.delete",
        name="delete",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def delete(
        self,
        context: AuthedServiceContext,
        uid: UID,
        force: bool = False,
    ) -> Union[SyftSuccess, SyftError]:
        worker = self._get_worker(context=context, uid=uid)
        if isinstance(worker, SyftError):
            return worker

        worker_pool_name = worker.worker_pool_name

        # relative
        from .worker_pool_service import SyftWorkerPoolService

        worker_pool_service: SyftWorkerPoolService = context.node.get_service(
            "SyftWorkerPoolService"
        )
        worker_pool_stash = worker_pool_service.stash
        result = worker_pool_stash.get_by_name(
            credentials=context.credentials, pool_name=worker.worker_pool_name
        )

        if result.is_err():
            return SyftError(
                f"Failed to retrieved WorkerPool {worker_pool_name} "
                f"associated with SyftWorker {uid}"
            )

        worker_pool = result.ok()
        if worker_pool is None:
            return SyftError(
                f"Failed to retrieved WorkerPool {worker_pool_name} "
                f"associated with SyftWorker {uid}"
            )

        if IN_KUBERNETES:
            # Kubernetes will only restart the worker NOT REMOVE IT
            runner = KubernetesRunner()
            runner.delete_pod(pod_name=worker.name)
            return SyftSuccess(
                # pod deletion is not supported in Kubernetes, removing and recreating the pod.
                message=(
                    "Worker deletion is not supported in Kubernetes. "
                    f"Removing and re-creating worker id={worker.id}"
                )
            )
        elif not context.node.in_memory_workers:
            # delete the worker using docker client sdk
            with contextlib.closing(docker.from_env()) as client:
                docker_container = _get_worker_container(client, worker)
                if isinstance(docker_container, SyftError):
                    return docker_container

                stopped = _stop_worker_container(worker, docker_container, force)
                if stopped is not None:
                    return stopped

        # remove the worker from the pool
        try:
            worker_linked_object = next(
                obj for obj in worker_pool.worker_list if obj.object_uid == uid
            )
            worker_pool.worker_list.remove(worker_linked_object)
        except StopIteration:
            pass

        # Delete worker from worker stash
        result = self.stash.delete_by_uid(credentials=context.credentials, uid=uid)
        if result.is_err():
            return SyftError(message=f"Failed to delete worker with uid: {uid}")

        # Update worker pool
        result = worker_pool_stash.update(context.credentials, obj=worker_pool)
        if result.is_err():
            return SyftError(message=f"Failed to update worker pool: {result.err()}")

        return SyftSuccess(
            message=f"Worker with id: {uid} deleted successfully from pool: {worker_pool.name}"
        )

    def _get_worker(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftWorker, SyftError]:
        result = self.stash.get_by_uid(credentials=context.credentials, uid=uid)
        if result.is_err():
            return SyftError(message=f"Failed to retrieve worker with UID {uid}")

        worker = result.ok()
        if worker is None:
            return SyftError(message=f"Worker does not exist for UID {uid}")

        return worker


def refresh_worker_status(
    workers: List[SyftWorker],
    worker_stash: WorkerStash,
    credentials: SyftVerifyKey,
) -> List[SyftWorker]:
    if IN_KUBERNETES:
        result = refresh_status_kubernetes(workers)
    else:
        result = refresh_status_docker(workers)

    if isinstance(result, SyftError):
        return result

    for worker in result:
        stash_result = worker_stash.update(
            credentials=credentials,
            obj=worker,
        )
        if stash_result.is_err():
            return SyftError(
                message=f"Failed to update status for worker: {worker.id}. Error: {stash_result.err()}"
            )

    return result


def refresh_status_kubernetes(workers: List[SyftWorker]) -> List[SyftWorker]:
    updated_workers = []
    runner = KubernetesRunner()
    for worker in workers:
        status = runner.get_pod_status(pod_name=worker.name)
        if not status:
            return SyftError(message=f"Pod does not exist. name={worker.name}")
        status, health, _ = map_pod_to_worker_status(status)
        worker.status = status
        worker.healthcheck = health
        updated_workers.append(worker)

    return updated_workers


def refresh_status_docker(workers: List[SyftWorker]) -> List[SyftWorker]:
    updated_workers = []

    with contextlib.closing(docker.from_env()) as client:
        for worker in workers:
            status = _get_worker_container_status(client, worker)
            if isinstance(status, SyftError):
                return status
            worker.status = status
            worker.healthcheck = _get_healthcheck_based_on_status(status=status)
            updated_workers.append(worker)

    return updated_workers


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
        return None
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
