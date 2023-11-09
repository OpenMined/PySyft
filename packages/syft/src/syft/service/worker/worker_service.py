# stdlib
import socket
from typing import List
from typing import Union

# third party
import docker

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


@serializable()
class DockerWorker(SyftObject):
    # version
    __canonical_name__ = "ContainerImage"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["container_id"]
    __attr_unique__ = ["container_id"]
    __repr_attrs__ = ["container_id", "created_at"]

    container_id: str
    created_at: DateTime


@instrument
@serializable()
class WorkerStash(BaseUIDStoreStash):
    object_type = DockerWorker
    settings: PartitionSettings = PartitionSettings(
        name=DockerWorker.__canonical_name__, object_type=DockerWorker
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)


# def get_default_env_vars(context: AuthedServiceContext):
#     if context.node.runs_in_docker:
#         # get env vars from current environment
#         return dict(os.environ)
#     else:
#         # read env vars from .env file
#         env_path = f"{context.node.host_syft_location}/packages/grid/.env"
#         with open(env_path) as f:
#             lines = f.read().splitlines()

#         default_env_vars = {}
#         for line in lines:
#             if "=" in line:
#                 try:
#                     var_name, value = line.split("=", 1)

#                     def remove_redundant_quotes(value):
#                         for s in ['"', "'"]:
#                             if len(value) != 0:
#                                 if value[0] == s:
#                                     value = value[1:]
#                                 if value[-1] == s:
#                                     value = value[:-1]

#                     value = remove_redundant_quotes(value)
#                     default_env_vars[var_name] = value
#                 except Exception as e:
#                     print("error parsing env file", e)
#         return default_env_vars


# PORT_COUNTER = 0


# def get_env_vars(context: AuthedServiceContext):
#     default_env_vars = get_default_env_vars(context)
#     # stdlib
#     import secrets

#     worker_tag = "".join([str(secrets.choice(list(range(10)))) for i in range(10)])
#     node = context.node
#     # TODO, improve
#     global PORT_COUNTER
#     PORT_COUNTER += 1
#     extra_env_vars = {
#         "SERVICE_NAME": "backend",
#         "CREATE_PRODUCER": "false",
#         "N_CONSUMERS": "1",
#         "DEV_MODE": node.dev_mode,
#         "DEFAULT_ROOT_USERNAME": f"worker-{worker_tag}",
#         "PORT": str(8003 + PORT_COUNTER),
#         "QUEUE_PORT": node.queue_config.client_config.queue_port,
#         "HTTP_PORT": str(88 + PORT_COUNTER),
#         "HTTPS_PORT": str(446 + PORT_COUNTER),
#         "DEFAULT_ROOT_EMAIL": f"{worker_tag}@openmined.org",
#     }
#     # if node.dev_mode:
#     #     extra_env_vars["WATCHFILES_FORCE_POLLING"] = "true"

#     result = {**default_env_vars, **extra_env_vars}
#     result.pop("NODE_PRIVATE_KEY", None)
#     return result


def get_main_backend() -> str:
    hostname = socket.gethostname()
    return f"{hostname}-backend-1"


def start_worker_container(worker_num: int, context: AuthedServiceContext):
    client = docker.from_env()
    existing_container_name = get_main_backend()
    hostname = socket.gethostname()
    worker_name = f"{hostname}-worker-{worker_num}"
    return create_new_container_from_existing(
        worker_name=worker_name,
        client=client,
        existing_container_name=existing_container_name,
    )


def create_new_container_from_existing(
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
            res = start_worker_container(WORKER_NUM, context)
            obj = DockerWorker(container_id=res.id, created_at=DateTime.now())
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
        workers: Union[List[DockerWorker], DockerWorker],
    ) -> Union[SyftSuccess, SyftError]:
        # listify
        if isinstance(workers, DockerWorker):
            workers = [workers]

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
