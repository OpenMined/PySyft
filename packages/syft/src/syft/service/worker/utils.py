# stdlib
import contextlib
import sys
from typing import List

# third party
import docker

# relative
from ..response import SyftError
from .worker_image import SyftWorkerImage
from .worker_pool import ContainerSpawnStatus
from .worker_pool import SyftWorker
from .worker_pool import WorkerHealth
from .worker_pool import WorkerOrchestrationType
from .worker_pool import WorkerStatus
from .worker_pool import _get_healthcheck_based_on_status


def run_container_using_docker(
    client: docker.DockerClient,
    worker_image: SyftWorkerImage,
    worker_name: str,
) -> ContainerSpawnStatus:
    # start container
    container = None
    error_message = None
    worker = None
    try:
        container = client.containers.run(
            worker_image.image_identifier.repo_with_tag,
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
            image=worker_image,
        )
    except Exception as e:
        error_message = f"Failed to run command in container. {worker_name} {worker_image}. {e}. {sys.stderr}"
        if container:
            worker = SyftWorker(
                name=worker_name,
                container_id=container.id,
                image_hash=container.image.id,
                status=WorkerStatus.STOPPED,
                healthcheck=WorkerHealth.UNHEALTHY,
                image=worker_image,
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
    results = []
    if not orchestration == WorkerOrchestrationType.DOCKER:
        return SyftError(message="Only Orchestration via Docker is supported.")

    with contextlib.closing(docker.from_env()) as client:
        for worker_count in range(1, number + 1):
            worker_name = f"{pool_name}-{worker_count}"
            spawn_result = run_container_using_docker(
                client=client,
                worker_image=worker_image,
                worker_name=worker_name,
            )
            results.append(spawn_result)

    return results
