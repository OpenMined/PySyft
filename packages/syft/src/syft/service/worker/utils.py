# stdlib
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


def run_container_using_docker(
    image_tag: SyftWorkerImageTag,
    worker_name: str,
) -> ContainerSpawnStatus:
    client = docker.from_env()

    # start container
    container = None
    error_message = None
    try:
        container = client.containers.run(
            image_tag.full_tag,
            name=worker_name,
            detach=True,
            auto_remove=True,
        )

        status = (
            WorkerStatus.STOPPED
            if container.status == "exited"
            else WorkerStatus.PENDING
        )
    except Exception as e:
        status = WorkerStatus.STOPPED
        error_message = f"Failed to run command in container. {worker_name} {image_tag}. {e}. {sys.stderr}"
        if container:
            container.stop()

    worker = SyftWorker(
        name=worker_name,
        container_id=container.id,
        image_hash=container.image.id,
        status=status,
    )

    return ContainerSpawnStatus(worker=worker, error=error_message)


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
        spawn_result = run_container_using_docker(worker_name, image_tag=image_tag)
        results.append(spawn_result)

    return results
