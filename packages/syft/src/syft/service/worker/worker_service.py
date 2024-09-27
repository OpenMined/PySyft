# stdlib
import contextlib
from typing import Any
from typing import cast

# third party
import docker
from docker.models.containers import Container

# relative
from ...custom_worker.k8s import IN_KUBERNETES
from ...custom_worker.k8s import PodStatus
from ...custom_worker.runner_k8s import KubernetesRunner
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.db import DBManager
from ...store.document_store_errors import StashException
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.uid import UID
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import AuthedServiceContext
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


@serializable(canonical_name="WorkerService", version=1)
class WorkerService(AbstractService):
    stash: WorkerStash

    def __init__(self, store: DBManager) -> None:
        self.stash = WorkerStash(store=store)

    @service_method(
        path="worker.start_workers",
        name="start_workers",
        roles=ADMIN_ROLE_LEVEL,
    )
    def start_workers(
        self, context: AuthedServiceContext, n: int = 1
    ) -> list[ContainerSpawnStatus]:
        """Add a Container Image."""

        return context.server.services.syft_worker_pool.add_workers(
            context, number=n, pool_name=DEFAULT_WORKER_POOL_NAME
        )

    @service_method(
        path="worker.get_all", name="get_all", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def list(self, context: AuthedServiceContext) -> list[SyftWorker]:
        """List all the workers."""
        workers = self.stash.get_all(context.credentials).unwrap()

        if context.server is not None and context.server.in_memory_workers:
            return workers
        else:
            # If container workers, check their statuses
            workers = refresh_worker_status(
                workers, self.stash, context.as_root_context().credentials
            ).unwrap()
        return workers

    @service_method(
        path="worker.status", name="status", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def status(
        self,
        context: AuthedServiceContext,
        uid: UID,
    ) -> tuple[WorkerStatus, WorkerHealth | None]:
        result = self.get(context=context, uid=uid)
        return result.status, result.healthcheck

    @service_method(
        path="worker.get",
        name="get",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get(self, context: AuthedServiceContext, uid: UID) -> SyftWorker:
        worker = self._get_worker(context=context, uid=uid).unwrap()

        if context.server is not None and context.server.in_memory_workers:
            return worker
        else:
            workers = refresh_worker_status(
                [worker], self.stash, context.as_root_context().credentials
            ).unwrap()
            return workers[0]

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
    ) -> bytes | str:
        worker = self._get_worker(context=context, uid=uid).unwrap()

        if context.server is not None and context.server.in_memory_workers:
            logs = b"Logs not implemented for In Memory Workers"
        elif IN_KUBERNETES:
            runner = KubernetesRunner()
            return runner.get_pod_logs(pod_name=worker.name)
        else:
            with contextlib.closing(docker.from_env()) as client:
                docker_container = _get_worker_container(client, worker).unwrap()
                try:
                    logs = cast(bytes, docker_container.logs())
                except docker.errors.APIError as e:
                    raise SyftException(
                        public_message=f"Failed to get worker {worker.id} container logs. Error {e}"
                    )

        return logs if raw else logs.decode(errors="ignore")

    def _delete(
        self, context: AuthedServiceContext, worker: SyftWorker, force: bool = False
    ) -> SyftSuccess:
        uid = worker.id
        if force and worker.job_id is not None:
            context.server.services.job.kill(context=context, id=worker.job_id)

        worker_pool_stash = context.server.services.syft_worker_pool.stash
        worker_pool = worker_pool_stash.get_by_name(
            credentials=context.credentials, pool_name=worker.worker_pool_name
        ).unwrap()

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
        elif not context.server.in_memory_workers:
            # delete the worker using docker client sdk
            with contextlib.closing(docker.from_env()) as client:
                docker_container = _get_worker_container(client, worker).unwrap()
                _stop_worker_container(worker, docker_container, force=force).unwrap()
        else:
            # kill the in memory worker thread
            context.server.remove_consumer_with_id(syft_worker_id=worker.id)

        # remove the worker from the pool
        try:
            worker_linked_object = next(
                obj for obj in worker_pool.worker_list if obj.object_uid == uid
            )
            worker_pool.worker_list.remove(worker_linked_object)
        except StopIteration:
            pass

        # Delete worker from worker stash
        self.stash.delete_by_uid(credentials=context.credentials, uid=uid).unwrap()

        # Update worker pool
        worker_pool_stash.update(context.credentials, obj=worker_pool).unwrap()

        return SyftSuccess(
            message=f"Worker with id: {uid} deleted successfully from pool: {worker_pool.name}"
        )

    @service_method(
        path="worker.delete",
        name="delete",
        roles=DATA_OWNER_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def delete(
        self,
        context: AuthedServiceContext,
        uid: UID,
        force: bool = False,
    ) -> SyftSuccess:
        worker = self._get_worker(context=context, uid=uid).unwrap()
        worker.to_be_deleted = True

        self.stash.update(context.credentials, worker).unwrap()
        if not force:
            # relative
            return SyftSuccess(message=f"Worker {uid} has been marked for deletion.")

        return self._delete(context, worker, force=True)

    @as_result(SyftException, StashException)
    def _get_worker(self, context: AuthedServiceContext, uid: UID) -> SyftWorker:
        return self.stash.get_by_uid(credentials=context.credentials, uid=uid).unwrap()


@as_result(SyftException)
def refresh_worker_status(
    workers: list[SyftWorker],
    worker_stash: WorkerStash,
    credentials: SyftVerifyKey,
) -> list[SyftWorker]:
    if IN_KUBERNETES:
        workers = refresh_status_kubernetes(workers).unwrap()
    else:
        workers = refresh_status_docker(workers).unwrap()

    for worker in workers:
        worker_stash.update(
            credentials=credentials,
            obj=worker,
        ).unwrap()

    return workers


@as_result(SyftException)
def refresh_status_kubernetes(workers: list[SyftWorker]) -> list[SyftWorker]:
    updated_workers = []
    runner = KubernetesRunner()
    for worker in workers:
        status: PodStatus | WorkerStatus | None = runner.get_pod_status(pod=worker.name)
        if not status:
            worker.status = WorkerStatus.STOPPED
            worker.healthcheck = WorkerHealth.UNHEALTHY
        else:
            status, health, _ = map_pod_to_worker_status(status)
            worker.status = status
            worker.healthcheck = health
            updated_workers.append(worker)

    return updated_workers


@as_result(SyftException)
def refresh_status_docker(workers: list[SyftWorker]) -> list[SyftWorker]:
    updated_workers = []
    with contextlib.closing(docker.from_env()) as client:
        for worker in workers:
            status = _get_worker_container_status(client, worker).unwrap()
            worker.status = status
            worker.healthcheck = _get_healthcheck_based_on_status(status=status)
            updated_workers.append(worker)
    return updated_workers


@as_result(SyftException)
def _stop_worker_container(
    worker: SyftWorker,
    container: Container,
    force: bool,
) -> None:
    try:
        # stop the container
        container.stop()
        # Remove the container and its volumes
        _remove_worker_container(container, force=force, v=True)
        return None
    except Exception as e:
        raise SyftException(
            public_message=f"Failed to delete worker with id: {worker.id}. Error: {e}"
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
