# stdlib
import socket
import sys
from typing import List

# third party
import docker

# relative
from ..response import SyftError
from .worker_image import SyftWorkerImage
from .worker_image import SyftWorkerImageTag
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

    # start container
    container = None
    error_message = None
    worker = None
    try:
        existing_container = client.containers.get(existing_container_name)
        network_mode = (f"container:{existing_container.id}",)

        container = client.containers.run(
            image_tag.full_tag,
            name=worker_name,
            detach=True,
            auto_remove=True,
            network_mode=network_mode,
            environment=environment,
        )

        status = (
            WorkerStatus.STOPPED
            if container.status == "exited"
            else WorkerStatus.PENDING
        )
        worker = SyftWorker(
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
        )
        results.append(spawn_result)

    return results
