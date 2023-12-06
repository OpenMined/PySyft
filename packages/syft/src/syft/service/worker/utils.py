# stdlib
from typing import List
from typing import Union

# relative
from ..response import SyftError
from .worker_image import SyftWorkerImage
from .worker_image import SyftWorkerImageTag
from .worker_pool import SyftWorker
from .worker_pool import WorkerOrchestrationType


def run_container_using_docker(
    worker_name: str,
    image_tag: SyftWorkerImageTag,
) -> Union[SyftError, SyftWorker]:
    pass


def run_containers(
    pool_name: str, worker_image: SyftWorkerImage, number: int, orchestration: str
) -> List[Union[SyftError, SyftWorker]]:
    image_tag = worker_image.image_tag

    results = []
    if not orchestration.lower() == WorkerOrchestrationType.DOCKER.value:
        return SyftError(message="Only Orchestration via Docker is supported.")

    for worker_count in range(1, number + 1):
        worker_name = f"{pool_name}-{worker_count}"
        run_result = run_container_using_docker(worker_name, image_tag=image_tag)
        results.append(run_result)

    return results
