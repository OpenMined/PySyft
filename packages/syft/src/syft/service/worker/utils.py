# stdlib
import contextlib
import json
import os
import socket
import socketserver
import sys
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

# third party
import docker
from pydantic import BaseModel

# relative
from ...abstract_node import AbstractNode
from ...custom_worker.builder import CustomWorkerBuilder
from ...custom_worker.config import DockerWorkerConfig
from ...node.credentials import SyftVerifyKey
from ...types.uid import UID
from ...util.util import get_queue_address
from ..response import SyftError
from .worker_image import SyftWorkerImage
from .worker_image_stash import SyftWorkerImageStash
from .worker_pool import ContainerSpawnStatus
from .worker_pool import SyftWorker
from .worker_pool import WorkerHealth
from .worker_pool import WorkerOrchestrationType
from .worker_pool import WorkerStatus

DEFAULT_WORKER_IMAGE_TAG = "openmined/default-worker-image-cpu:0.0.1"
DEFAULT_WORKER_POOL_NAME = "default-pool"


class ImageBuildResult(BaseModel):
    image_hash: str
    logs: Iterable[str]


def backend_container_name() -> str:
    hostname = socket.gethostname()
    service_name = os.getenv("SERVICE", "backend")
    return f"{hostname}-{service_name}-1"


def get_container(docker_client: docker.DockerClient, container_name: str):
    try:
        existing_container = docker_client.containers.get(container_name)
    except docker.errors.NotFound:
        existing_container = None

    return existing_container


def extract_config_from_backend(worker_name: str, docker_client: docker.DockerClient):
    # Existing main backend container
    backend_container = get_container(
        docker_client, container_name=backend_container_name()
    )

    # Config with defaults
    extracted_config = {"volume_binds": {}, "network_mode": None, "environment": {}}

    if backend_container is None:
        return extracted_config

    # Inspect the existing container
    details = backend_container.attrs

    host_config = details["HostConfig"]
    environment = details["Config"]["Env"]

    # Extract Volume Binds
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
        extracted_config["volume_binds"][key] = {"bind": bind, "mode": mode}

    # Extract Environment Variables
    extracted_config["environment"] = dict([e.split("=", 1) for e in environment])
    extracted_config["network_mode"] = f"container:{backend_container.id}"

    return extracted_config


def get_free_tcp_port():
    with socketserver.TCPServer(("localhost", 0), None) as s:
        free_port = s.server_address[1]
        return free_port


def run_container_using_docker(
    worker_image: SyftWorkerImage,
    docker_client: docker.DockerClient,
    worker_name: str,
    worker_count: int,
    pool_name: str,
    queue_port: int,
    debug: bool = False,
    username: Optional[str] = None,
    password: Optional[str] = None,
    registry_url: Optional[str] = None,
) -> ContainerSpawnStatus:
    if not worker_image.is_built:
        raise Exception("Image must be built before running it.")

    # Get hostname
    hostname = socket.gethostname()

    # Create a random UID for Worker
    syft_worker_uid = UID()

    # container name
    container_name = f"{hostname}-{worker_name}"

    # start container
    container = None
    error_message = None
    worker = None
    try:
        # login to the registry through the client
        # so that the subsequent pull/run commands work
        if registry_url and username and password:
            docker_client.login(
                username=username,
                password=password,
                registry=registry_url,
            )

        # If container with given name already exists then stop it
        # and recreate it.
        existing_container = get_container(
            docker_client=docker_client,
            container_name=container_name,
        )
        if existing_container:
            existing_container.remove(force=True)

        # Extract Config from backend container
        backend_host_config = extract_config_from_backend(
            worker_name=worker_name, docker_client=docker_client
        )

        # Create List of Envs to pass
        environment = backend_host_config["environment"]
        environment["CREATE_PRODUCER"] = "false"
        environment["N_CONSUMERS"] = 1
        environment["PORT"] = str(get_free_tcp_port())
        environment["HTTP_PORT"] = str(88 + worker_count)
        environment["HTTPS_PORT"] = str(446 + worker_count)
        environment["CONSUMER_SERVICE_NAME"] = pool_name
        environment["SYFT_WORKER_UID"] = syft_worker_uid
        environment["DEV_MODE"] = debug
        environment["QUEUE_PORT"] = queue_port
        environment["CONTAINER_HOST"] = "docker"

        container = docker_client.containers.run(
            worker_image.image_identifier.full_name_with_tag,
            name=f"{hostname}-{worker_name}",
            detach=True,
            auto_remove=True,
            network_mode=backend_host_config["network_mode"],
            environment=environment,
            volumes=backend_host_config["volume_binds"],
            tty=True,
            stdin_open=True,
            labels={"orgs.openmined.syft": "this is a syft worker container"},
        )

        status = (
            WorkerStatus.STOPPED
            if container.status == "exited"
            else WorkerStatus.PENDING
        )

        healthcheck: WorkerHealth = _get_healthcheck_based_on_status(status)

        worker = SyftWorker(
            id=syft_worker_uid,
            name=worker_name,
            container_id=container.id,
            status=status,
            healthcheck=healthcheck,
            image=worker_image,
            worker_pool_name=pool_name,
        )
    except Exception as e:
        error_message = f"Failed to run command in container. {worker_name} {worker_image}. {e}. {sys.stderr}"
        if container:
            worker = SyftWorker(
                name=worker_name,
                container_id=container.id,
                status=WorkerStatus.STOPPED,
                healthcheck=WorkerHealth.UNHEALTHY,
                image=worker_image,
                worker_pool_name=pool_name,
            )
            container.stop()

    return ContainerSpawnStatus(
        worker_name=worker_name, worker=worker, error=error_message
    )


def run_workers_in_threads(
    node: AbstractNode,
    pool_name: str,
    number: int,
    start_idx: int = 0,
) -> List[ContainerSpawnStatus]:
    results = []

    for worker_count in range(start_idx + 1, number + 1):
        error = None
        worker_name = f"{pool_name}-{worker_count}"
        worker = SyftWorker(
            name=worker_name,
            status=WorkerStatus.RUNNING,
            worker_pool_name=pool_name,
            healthcheck=WorkerHealth.HEALTHY,
        )
        try:
            port = node.queue_config.client_config.queue_port
            address = get_queue_address(port)
            node.add_consumer_for_service(
                service_name=pool_name,
                syft_worker_id=worker.id,
                address=address,
            )
        except Exception as e:
            print(
                f"Failed to start consumer for Pool Name: {pool_name}, Worker Name: {worker_name}"
            )
            worker.status = WorkerStatus.STOPPED
            error = str(e)

        container_status = ContainerSpawnStatus(
            worker_name=worker_name,
            worker=worker,
            error=error,
        )

        results.append(container_status)

    return results


def run_containers(
    pool_name: str,
    worker_image: SyftWorkerImage,
    number: int,
    orchestration: WorkerOrchestrationType,
    queue_port: int,
    dev_mode: bool = False,
    start_idx: int = 0,
    username: Optional[str] = None,
    password: Optional[str] = None,
    registry_url: Optional[str] = None,
) -> List[ContainerSpawnStatus]:
    results = []

    if orchestration not in [WorkerOrchestrationType.DOCKER]:
        return SyftError(message="Only Orchestration via Docker is supported.")

    if not worker_image.is_built:
        return SyftError(message="Image must be built before running it.")

    with contextlib.closing(docker.from_env()) as client:
        for worker_count in range(start_idx + 1, number + 1):
            worker_name = f"{pool_name}-{worker_count}"
            spawn_result = run_container_using_docker(
                docker_client=client,
                worker_name=worker_name,
                worker_count=worker_count,
                worker_image=worker_image,
                pool_name=pool_name,
                queue_port=queue_port,
                debug=dev_mode,
                username=username,
                password=password,
                registry_url=registry_url,
            )
            results.append(spawn_result)

    return results


def create_default_image(
    credentials: SyftVerifyKey,
    image_stash: SyftWorkerImageStash,
    dev_mode: bool,
    syft_version_tag: str,
) -> Union[SyftError, SyftWorkerImage]:
    # TODO: Hardcode worker dockerfile since not able to COPY
    # worker_cpu.dockerfile to backend in backend.dockerfile.

    # default_cpu_dockerfile = get_syft_cpu_dockerfile()
    # DockerWorkerConfig.from_path(default_cpu_dockerfile)

    default_cpu_dockerfile = f"""ARG SYFT_VERSION_TAG='{syft_version_tag}' \n"""
    default_cpu_dockerfile += """FROM openmined/grid-backend:${SYFT_VERSION_TAG}
    ARG PYTHON_VERSION="3.11"
    ARG SYSTEM_PACKAGES=""
    ARG PIP_PACKAGES="pip --dry-run"
    ARG CUSTOM_CMD='echo "No custom commands passed"'

    # Worker specific environment variables go here
    ENV SYFT_WORKER="true"
    ENV DOCKER_TAG=${SYFT_VERSION_TAG}

    RUN apk update && \
        apk add ${SYSTEM_PACKAGES} && \
        pip install --user ${PIP_PACKAGES} && \
        bash -c "$CUSTOM_CMD"
    """
    worker_config = DockerWorkerConfig(dockerfile=default_cpu_dockerfile)

    result = image_stash.get_by_docker_config(
        credentials=credentials,
        config=worker_config,
    )

    if result.ok() is None:
        default_syft_image = SyftWorkerImage(
            config=worker_config,
            created_by=credentials,
        )
        result = image_stash.set(credentials, default_syft_image)

        if result.is_err():
            return SyftError(message=f"Failed to save image stash: {result.err()}")

    default_syft_image = result.ok()

    return default_syft_image


def _get_healthcheck_based_on_status(status: WorkerStatus) -> WorkerHealth:
    if status in [WorkerStatus.PENDING, WorkerStatus.RUNNING]:
        return WorkerHealth.HEALTHY
    else:
        return WorkerHealth.UNHEALTHY


def docker_build(
    image: SyftWorkerImage, **kwargs
) -> Union[ImageBuildResult, SyftError]:
    try:
        builder = CustomWorkerBuilder()
        (built_image, logs) = builder.build_image(
            config=image.config,
            tag=image.image_identifier.full_name_with_tag,
            rm=True,
            forcerm=True,
            **kwargs,
        )
        return ImageBuildResult(image_hash=built_image.id, logs=parse_output(logs))
    except docker.errors.APIError as e:
        return SyftError(
            message=f"Docker API error when building {image.image_identifier}. Reason - {e}"
        )
    except docker.errors.DockerException as e:
        return SyftError(
            message=f"Docker exception when building {image.image_identifier}. Reason - {e}"
        )
    except Exception as e:
        return SyftError(
            message=f"Unknown exception when building {image.image_identifier}. Reason - {e}"
        )


def docker_push(
    image: SyftWorkerImage,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Union[List[str], SyftError]:
    try:
        builder = CustomWorkerBuilder()
        result = builder.push_image(
            # this should be consistent with docker build command
            tag=image.image_identifier.full_name_with_tag,
            registry_url=image.image_identifier.registry_host,
            username=username,
            password=password,
        )

        if "error" in result:
            return SyftError(
                message=f"Failed to push {image.image_identifier}. Logs - {result}"
            )

        return result.split(os.linesep)
    except docker.errors.APIError as e:
        return SyftError(
            message=f"Docker API error when pushing {image.image_identifier}. {e}"
        )
    except docker.errors.DockerException as e:
        return SyftError(
            message=f"Docker exception when pushing {image.image_identifier}. Reason - {e}"
        )
    except Exception as e:
        return SyftError(
            message=f"Unknown exception when pushing {image.image_identifier}. Reason - {e}"
        )


def parse_output(log_iterator: Iterable) -> str:
    log = ""
    for line in log_iterator:
        for item in line.values():
            if isinstance(item, str):
                log += item
            elif isinstance(item, dict):
                log += json.dumps(item)
            else:
                log += str(item)
    return log
