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
from .worker_pool import WorkerHealth
from .worker_pool import WorkerOrchestrationType
from .worker_pool import WorkerStatus
from .worker_pool import _get_healthcheck_based_on_status


def run_container_using_docker(
    client: docker.DockerClient,
    image_tag: SyftWorkerImageTag,
    worker_name: str,
    full_tag: str,
) -> ContainerSpawnStatus:
    # start container
    container = None
    error_message = None
    worker = None
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

        healthcheck: WorkerHealth = _get_healthcheck_based_on_status(status)

        worker = SyftWorker(
            name=worker_name,
            container_id=container.id,
            image_hash=container.image.id,
            status=status,
            healthcheck=healthcheck,
            full_image_tag=full_tag,
        )
    except Exception as e:
        error_message = f"Failed to run command in container. {worker_name} {image_tag}. {e}. {sys.stderr}"
        if container:
            worker = SyftWorker(
                name=worker_name,
                container_id=container.id,
                image_hash=container.image.id,
                status=WorkerStatus.STOPPED,
                healthcheck=WorkerHealth.UNHEALTHY,
                full_image_tag=full_tag,
            )
            container.stop()

    return ContainerSpawnStatus(
        worker_name=worker_name, worker=worker, error=error_message
    )


def run_containers(
    client: docker.DockerClient,
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
            client=client,
            worker_name=worker_name,
            image_tag=image_tag,
            full_tag=worker_image.full_tag,
        )
        results.append(spawn_result)

    return results
