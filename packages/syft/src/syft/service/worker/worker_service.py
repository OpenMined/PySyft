# stdlib
import copy
import os
import socket
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import docker
import kr8s
from kr8s.objects import Pod

# relative
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...store.document_store import SYFT_OBJECT_VERSION_1
from ...store.document_store import SyftObject
from ...store.document_store import SyftSuccess
from ...types.datetime import DateTime
from ...util.telemetry import instrument
from ..service import AbstractService
from ..service import AuthedServiceContext
from ..service import SyftError
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL

container_host = os.getenv("CONTAINER_HOST", "docker")
ContainerType = Union[docker.models.containers.Container, Pod]


@serializable()
class ContainerWorker(SyftObject):
    # version
    __canonical_name__ = "ContainerWorker"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["container_id"]
    __attr_unique__ = ["container_id"]
    __repr_attrs__ = ["container_id", "created_at"]

    container_id: Optional[str]
    created_at: DateTime
    name: str
    container_host: str


@instrument
@serializable()
class WorkerStash(BaseUIDStoreStash):
    object_type = ContainerWorker
    settings: PartitionSettings = PartitionSettings(
        name=ContainerWorker.__canonical_name__, object_type=ContainerWorker
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)


def get_main_backend_docker() -> str:
    hostname = socket.gethostname()
    return f"{hostname}-backend-1"


def get_main_backend_k8s() -> str:
    return "backend-0"


def start_worker_container(
    worker_num: int, context: AuthedServiceContext
) -> Tuple[str, ContainerType]:
    if container_host == "docker":
        client = docker.from_env()
        existing_container_name = get_main_backend_docker()
        hostname = socket.gethostname()
        worker_name = f"{hostname}-worker-{worker_num}"
        container = create_new_container_from_existing_docker(
            worker_name=worker_name,
            client=client,
            existing_container_name=existing_container_name,
        )
        return worker_name, container.id, container
    elif container_host == "k8s":
        client = client = kr8s.api()
        existing_container_name = get_main_backend_k8s()
        worker_name = f"worker-{worker_num}"
        pod = create_new_container_from_existing_k8s(
            worker_name=worker_name,
            client=client,
            existing_container_name=existing_container_name,
        )
        return worker_name, pod.metadata.uid, pod

    raise Exception(f"Can't start workers. Unknown container host {container_host}")


def create_new_container_from_existing_docker(
    worker_name: str, client: docker.client.DockerClient, existing_container_name: str
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
    environment["DEFAULT_ROOT_USERNAME"] = worker_name
    environment["DEFAULT_ROOT_EMAIL"] = f"{worker_name}@openmined.org"
    environment["PORT"] = str(8003 + WORKER_NUM)
    environment["HTTP_PORT"] = str(88 + WORKER_NUM)
    environment["HTTPS_PORT"] = str(446 + WORKER_NUM)
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


def create_new_container_from_existing_k8s(
    worker_name: str, client: kr8s.Api, existing_container_name: str
) -> kr8s.objects.Pod:
    # Get the existing pod
    existing_pod = Pod.get(existing_container_name)

    spec = existing_pod.spec
    image = spec.containers[0].image
    env = spec.containers[0].env
    new_env = copy.deepcopy(env)

    for item in new_env:
        if item["name"] == "N_CONSUMERS":
            item["value"] = "1"
        if item["name"] == "CREATE_PRODUCER":
            item["value"] = "False"
        if item["name"] == "DEFAULT_ROOT_USERNAME":
            item["value"] = worker_name
        if item["name"] == "DEFAULT_ROOT_EMAIL":
            item["value"] = f"{worker_name}@openmined.org"

    pod = Pod.gen(name=worker_name, image=image, env=new_env)
    pod.create()
    return pod


WORKER_NUM = 0


@instrument
@serializable()
class WorkerService(AbstractService):
    store: DocumentStore
    stash: WorkerStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = WorkerStash(store=store)

    @service_method(
        path="worker.start_workers", name="start_workers", roles=ADMIN_ROLE_LEVEL
    )
    def start_workers(
        self, context: AuthedServiceContext, n: int = 1
    ) -> Union[SyftSuccess, SyftError]:
        """Add a Container Image."""
        for _worker_num in range(n):
            global WORKER_NUM
            WORKER_NUM += 1
            worker_name, container_uid, container = start_worker_container(
                WORKER_NUM, context
            )
            obj = ContainerWorker(
                name=worker_name,
                container_id=container_uid,
                created_at=DateTime.now(),
                container_host=container_host,
            )
            result = self.stash.set(context.credentials, obj)
            if result.is_err():
                return SyftError(message=f"Failed to start worker. {result.err()}")

        return SyftSuccess(message=f"{n} workers added")

    @service_method(path="worker.list", name="list", roles=ADMIN_ROLE_LEVEL)
    def list(self, context: AuthedServiceContext) -> Union[SyftSuccess, SyftError]:
        """Add a Container Image."""
        result = self.stash.get_all(context.credentials)

        if result.is_err():
            return SyftError(message=f"Failed to fetch workers. {result.err()}")
        else:
            return result.ok()

    @service_method(path="worker.stop", name="stop", roles=ADMIN_ROLE_LEVEL)
    def stop(
        self,
        context: AuthedServiceContext,
        workers: Union[List[ContainerWorker], ContainerWorker],
    ) -> Union[SyftSuccess, SyftError]:
        # listify
        if isinstance(workers, ContainerWorker):
            workers = [workers]

        if container_host == "docker":
            client = docker.from_env()
            for w in workers:
                result = self.stash.delete_by_uid(context.credentials, uid=w.id)

                if result.is_err():
                    return SyftError(message=f"Failed to stop workers {result.err()}")

                # stop container
                try:
                    client.containers.list(filters={"id": w.container_id})[0].stop()
                    # also prune here?
                except Exception as e:
                    # we dont throw an error here because apparently the container was already killed
                    print(f"Failed to kill container {e}")

            return SyftSuccess(message=f"{len(workers)} workers stopped")
        elif container_host == "k8s":
            client = client = kr8s.api()
            for w in workers:
                result = self.stash.delete_by_uid(context.credentials, uid=w.id)

                if result.is_err():
                    return SyftError(message=f"Failed to stop workers {result.err()}")

                # stop container
                try:
                    pods = kr8s.get("pods", namespace=kr8s.ALL)
                    for pod in pods:
                        if pod.metadata.uid == w.container_id:
                            pod.delete()
                except Exception as e:
                    # we dont throw an error here because apparently the container was already killed
                    print(f"Failed to kill container {e}")

            return SyftSuccess(message=f"{len(workers)} workers stopped")

        raise Exception(f"Can't stop workers. Unknown container host {container_host}")
