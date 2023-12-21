# stdlib
import contextlib
import json
import os
import sys
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import docker

# relative
from ...custom_worker.builder import CustomWorkerBuilder
from ...custom_worker.builder import Image
from ..response import SyftError
from .worker_image import SyftWorkerImage
from .worker_image import SyftWorkerImageTag
from .worker_pool import ContainerSpawnStatus
from .worker_pool import SyftWorker
from .worker_pool import WorkerOrchestrationType
from .worker_pool import WorkerStatus


def docker_build(
    image: SyftWorkerImage
) -> Union[Tuple[Image, Iterable[str]], SyftError]:
    try:
        builder = CustomWorkerBuilder()
        (image, logs) = builder.build_image(
            config=image.config,
            tag=image.image_tag.full_tag,
        )
        parsed_logs = parse_output(logs)
        return (image, parsed_logs)
    except docker.errors.APIError as e:
        return SyftError(
            message=f"Docker API error when building {image.image_tag}. Reason - {e}"
        )
    except docker.errors.DockerException as e:
        return SyftError(
            message=f"Docker exception when building {image.image_tag}. Reason - {e}"
        )
    except Exception as e:
        return SyftError(
            message=f"Unknown exception when building {image.image_tag}. Reason - {e}"
        )


def docker_push(
    self,
    image: SyftWorkerImage,
    username: str = "",
    password: str = "",
) -> List[str]:
    try:
        builder = CustomWorkerBuilder()
        result = builder.push_image(
            tag=image.image_tag.full_tag,
            registry_url=image.image_tag.registry_host,
            username=username,
            password=password,
        )

        parsed_result = result.split(os.linesep)

        if "error" in result:
            result = SyftError(
                message=f"Failed to push {image.image_tag}. Logs - {parsed_result}"
            )

        return parsed_result
    except docker.errors.APIError as e:
        return SyftError(
            message=f"Docker API error when pushing {image.image_tag}. {e}"
        )
    except docker.errors.DockerException as e:
        return SyftError(
            message=f"Docker exception when pushing {image.image_tag}. Reason - {e}"
        )
    except Exception as e:
        return SyftError(
            message=f"Unknown exception when pushing {image.image_tag}. Reason - {e}"
        )


def parse_output(self, log_iter: Iterable) -> str:
    log = ""
    for line in log_iter:
        for item in line.values():
            if isinstance(item, str):
                log += item
            elif isinstance(item, dict):
                log += json.dumps(item)
            else:
                log += str(item)
    return log


def docker_run(
    client: docker.DockerClient,
    image_tag: SyftWorkerImageTag,
    worker_name: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    registry_url: Optional[str] = None,
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
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> List[ContainerSpawnStatus]:
    image_tag = worker_image.image_tag

    results = []
    if not orchestration == WorkerOrchestrationType.DOCKER:
        return SyftError(message="Only Orchestration via Docker is supported.")

    with contextlib.closing(docker.from_env()) as client:
        for worker_count in range(1, number + 1):
            worker_name = f"{pool_name}-{worker_count}"
            spawn_result = docker_run(
                client=client,
                worker_name=worker_name,
                image_tag=image_tag,
                registry_url=image_tag.registry,
                username=username,
                password=password,
            )
            results.append(spawn_result)

    return results
