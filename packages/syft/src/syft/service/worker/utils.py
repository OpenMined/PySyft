# stdlib
import contextlib
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
    client: docker.DockerClient,
    image_tag: SyftWorkerImageTag,
    worker_name: str,
    username=None,
    password=None,
    registry_url: str = None,
) -> ContainerSpawnStatus:
    # start container
    container = None
    error_message = None
    worker = None

    try:
        # login to the registry through the client
        # so that the subsequent pull/run commands work
        if registry_url and username and password:
            client.login(username=username, password=password)

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
    username: str = None,
    password: str = None,
) -> List[ContainerSpawnStatus]:
    image_tag = worker_image.image_tag

    results = []
    if not orchestration == WorkerOrchestrationType.DOCKER:
        return SyftError(message="Only Orchestration via Docker is supported.")

    with contextlib.closing(docker.from_env()) as client:
        for worker_count in range(1, number + 1):
            worker_name = f"{pool_name}-{worker_count}"
            spawn_result = run_container_using_docker(
                client=client,
                worker_name=worker_name,
                image_tag=image_tag,
                registry_url=image_tag.registry,
                username=username,
                password=password,
            )
            results.append(spawn_result)

    return results
