# stdlib
import socket
import sys
from typing import List

# third party
import docker

# relative
from ...custom_worker.config import DockerWorkerConfig
from ...node.credentials import SyftVerifyKey
from ...types.uid import UID
from ...util.util import get_syft_cpu_dockerfile
from ..response import SyftError
from .worker_image import SyftWorkerImage
from .worker_image import SyftWorkerImageTag
from .worker_image_stash import SyftWorkerImageStash
from .worker_pool import ContainerSpawnStatus
from .worker_pool import SyftWorker
from .worker_pool import WorkerOrchestrationType
from .worker_pool import WorkerStatus


def get_main_backend() -> str:
    hostname = socket.gethostname()
    return f"{hostname}-backend-1"


def run_container_using_docker(
    image_tag: SyftWorkerImageTag,
    worker_name: str,
    worker_count: int,
    pool_name: str,
    debug: bool = False,
) -> ContainerSpawnStatus:
    client = docker.from_env()

    # Existing main backend container
    existing_container_name = get_main_backend()
    syft_worker_uid = UID()

    # Create List of Envs to pass
    environment = {}
    environment["CREATE_PRODUCER"] = "false"
    environment["N_CONSUMERS"] = 1
    environment["DEFAULT_ROOT_USERNAME"] = worker_name
    environment["DEFAULT_ROOT_EMAIL"] = f"{worker_name}@openmined.org"
    environment["PORT"] = str(8003 + worker_count)
    environment["HTTP_PORT"] = str(88 + worker_count)
    environment["HTTPS_PORT"] = str(446 + worker_count)
    environment["CONSUMER_SERVICE_NAME"] = pool_name
    environment["SYFT_WORKER_UID"] = syft_worker_uid
    environment["DEV_MODE"] = debug

    # start container
    container = None
    error_message = None
    worker = None
    try:
        try:
            existing_container = client.containers.get(existing_container_name)
        except docker.errors.NotFound:
            existing_container = None

        network_mode = (
            f"container:{existing_container.id}" if existing_container else None
        )

        container = client.containers.run(
            image_tag.full_tag,
            name=worker_name,
            detach=True,
            auto_remove=True,
            network_mode=network_mode,
            environment=environment,
            tty=True,
            stdin_open=True,
        )

        status = (
            WorkerStatus.STOPPED
            if container.status == "exited"
            else WorkerStatus.PENDING
        )
        worker = SyftWorker(
            id=syft_worker_uid,
            name=worker_name,
            container_id=container.id,
            image_hash=container.image.id,
            status=status,
        )
    except Exception as e:
        error_message = f"Failed to run command in container. {worker_name} {image_tag}. {e}. {sys.stderr}"
        if container:
            worker = SyftWorker(
                name=worker_name,
                container_id=container.id,
                image_hash=container.image.id,
                status=WorkerStatus.STOPPED,
            )
            container.stop()

    return ContainerSpawnStatus(
        worker_name=worker_name, worker=worker, error=error_message
    )


def run_containers(
    pool_name: str,
    worker_image: SyftWorkerImage,
    number: int,
    orchestration: WorkerOrchestrationType,
    dev_mode: bool = False,
) -> List[ContainerSpawnStatus]:
    image_tag = worker_image.image_tag

    results = []
    if not orchestration == WorkerOrchestrationType.DOCKER:
        return SyftError(message="Only Orchestration via Docker is supported.")

    for worker_count in range(1, number + 1):
        worker_name = f"{pool_name}-{worker_count}"
        spawn_result = run_container_using_docker(
            worker_name=worker_name,
            worker_count=worker_count,
            image_tag=image_tag,
            pool_name=pool_name,
            debug=dev_mode,
        )
        results.append(spawn_result)

    return results


def create_default_image(credentials: SyftVerifyKey, image_stash: SyftWorkerImageStash):
    default_cpu_dockerfile = get_syft_cpu_dockerfile()
    worker_config = DockerWorkerConfig.from_path(default_cpu_dockerfile)

    result = image_stash.get_by_docker_config(credentials, worker_config)

    if result.ok() is None:
        default_syft_image = SyftWorkerImage(
            config=worker_config, created_by=credentials
        )
        result = image_stash.set(credentials, default_syft_image)

        if result.is_err():
            print(f"Failed to save image stash: {result.err()}")

    default_syft_image = result.ok()

    return default_syft_image


DEFAULT_WORKER_IMAGE_TAG = "openmined/default-worker-image-cpu:0.0.1"
DEFAULT_WORKER_POOL_NAME = "default"
