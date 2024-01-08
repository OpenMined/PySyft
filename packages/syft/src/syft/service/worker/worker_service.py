# stdlib
import contextlib
import socket
from typing import List
from typing import Tuple
from typing import Union

# third party
import docker
from result import Result

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
from .utils import DEFAULT_WORKER_POOL_NAME
from .utils import _get_healthcheck_based_on_status
from .worker_pool import ContainerSpawnStatus
from .worker_pool import SyftWorker
from .worker_pool import WorkerHealth
from .worker_pool import WorkerStatus
from .worker_pool import _get_worker_container_status
from .worker_stash import WorkerStash

WORKER_NUM = 0


def get_main_backend():
    hostname = socket.gethostname()
    return f"{hostname}-backend-1"


def start_worker_container(
    worker_num: int, context: AuthedServiceContext, syft_worker_uid
):
    client = docker.from_env()
    existing_container_name = get_main_backend()
    hostname = socket.gethostname()
    worker_name = f"{hostname}-worker-{worker_num}"
    return create_new_container_from_existing(
        worker_name=worker_name,
        client=client,
        existing_container_name=existing_container_name,
        syft_worker_uid=syft_worker_uid,
    )


def create_new_container_from_existing(
    worker_name: str,
    client: docker.client.DockerClient,
    existing_container_name: str,
    syft_worker_uid,
) -> docker.models.containers.Container:
    # Get the existing container
    existing_container = client.containers.get(existing_container_name)

    # Inspect the existing container
    details = existing_container.attrs

    # Extract relevant settings
    image = details["Config"]["Image"]
    command = details["Config"]["Cmd"]
    environment = details["Config"]["Env"]
    ports = details["NetworkSettings"]["Ports"]
    host_config = details["HostConfig"]

    volumes = {}
    for vol in host_config["Binds"]:
        parts = vol.split(":")
        key = parts[0]
        bind = parts[1]
        mode = parts[2]
        if "/storage" in bind:
            # we need this because otherwise we are using the same node private key
            # which will make account creation fail
            worker_postfix = worker_name.split("-", 1)[1]
            key = f"{key}-{worker_postfix}"
        volumes[key] = {"bind": bind, "mode": mode}

    # we need this because otherwise we are using the same node private key
    # which will make account creation fail

    environment = dict([e.split("=", 1) for e in environment])
    environment["CREATE_PRODUCER"] = "false"
    environment["N_CONSUMERS"] = 1
    environment["DOCKER_WORKER_NAME"] = worker_name
    environment["DEFAULT_ROOT_USERNAME"] = worker_name
    environment["DEFAULT_ROOT_EMAIL"] = f"{worker_name}@openmined.org"
    environment["PORT"] = str(8003 + WORKER_NUM)
    environment["HTTP_PORT"] = str(88 + WORKER_NUM)
    environment["HTTPS_PORT"] = str(446 + WORKER_NUM)
    environment["SYFT_WORKER_UID"] = str(syft_worker_uid)

    environment.pop("NODE_PRIVATE_KEY", None)

    new_container = client.containers.create(
        name=worker_name,
        image=image,
        command=command,
        environment=environment,
        ports=ports,
        detach=True,
        volumes=volumes,
        tty=True,
        stdin_open=True,
        network_mode=f"container:{existing_container.id}",
    )

    new_container.start()
    return new_container


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

    @service_method(path="worker.get_all", name="get_all", roles=ADMIN_ROLE_LEVEL)
    def list(self, context: AuthedServiceContext) -> Union[SyftSuccess, SyftError]:
        """List all the workers."""
        result = self.stash.get_all(context.credentials)

        if result.is_err():
            return SyftError(message=f"Failed to fetch workers. {result.err()}")

        workers = result.ok()

        if context.node.in_memory_workers:
            return workers

        # If container workers, check their statuses
        for idx, worker in enumerate(workers):
            result = check_and_update_status_for_worker(
                worker=worker,
                worker_stash=self.stash,
                credentials=context.credentials,
            )
            if result.is_err():
                return SyftError(
                    message=f"Failed to update status for worker: {worker.id}. Error: {result.err()}"
                )
            workers[idx] = worker

        return workers

    @service_method(path="worker.status", name="status", roles=ADMIN_ROLE_LEVEL)
    def status(
        self,
        context: AuthedServiceContext,
        uid: UID,
    ) -> Union[Tuple[WorkerStatus, WorkerHealth], SyftError]:
        result = self.stash.get_by_uid(credentials=context.credentials, uid=uid)
        if result.is_err():
            return SyftError(message=f"Failed to retrieve worker with UID {uid}")
        worker: SyftWorker = result.ok()

        if context.node.in_memory_workers:
            return worker.status, worker.healthcheck

        result = check_and_update_status_for_worker(
            worker=worker,
            worker_stash=self.stash,
            credentials=context.credentials,
        )

        if result.is_err():
            return SyftError(
                message=f"Failed to update status for worker: {worker.id}. Error: {result.err()}"
            )

        worker = result.ok()

        return worker.status, worker.healthcheck


def check_and_update_status_for_worker(
    worker: SyftWorker,
    worker_stash: WorkerStash,
    credentials: SyftVerifyKey,
) -> Result[SyftWorker, str]:
    with contextlib.closing(docker.from_env()) as client:
        worker_status = _get_worker_container_status(client, worker)

    if isinstance(worker_status, SyftError):
        return worker_status

    worker.status = worker_status

    worker.healthcheck = _get_healthcheck_based_on_status(status=worker_status)

    result = worker_stash.update(
        credentials=credentials,
        obj=worker,
    )

    return result
