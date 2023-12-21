# stdlib
import socket
from typing import Union

# third party
import docker

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.document_store import SyftSuccess
from ...util.telemetry import instrument
from ..service import AbstractService
from ..service import AuthedServiceContext
from ..service import SyftError
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from .worker_stash import WorkerStash


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


WORKER_NUM = 0


@instrument
@serializable()
class WorkerService(AbstractService):
    store: DocumentStore
    stash: WorkerStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = WorkerStash(store=store)

    # @service_method(
    #     path="worker.start_workers", name="start_workers", roles=ADMIN_ROLE_LEVEL
    # )
    # def start_workers(
    #     self, context: AuthedServiceContext, n: int = 1
    # ) -> Union[SyftSuccess, SyftError]:
    #     """Add a Container Image."""
    #     for _worker_num in range(n):
    #         global WORKER_NUM
    #         WORKER_NUM += 1
    #         res = start_worker_container(WORKER_NUM, context)
    #         obj = DockerWorker(container_id=res.id, created_at=DateTime.now())
    #         result = self.stash.set(context.credentials, obj)
    #         if result.is_err():
    #             return SyftError(message=f"Failed to start worker. {result.err()}")

    #     return SyftSuccess(message=f"{n} workers added")

    @service_method(path="worker.list", name="list", roles=ADMIN_ROLE_LEVEL)
    def list(self, context: AuthedServiceContext) -> Union[SyftSuccess, SyftError]:
        """Add a Container Image."""
        result = self.stash.get_all(context.credentials)

        if result.is_err():
            return SyftError(message=f"Failed to fetch workers. {result.err()}")
        else:
            return result.ok()
