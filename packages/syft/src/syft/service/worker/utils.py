# stdlib
import contextlib
import socket
import sys
from typing import List

# third party
import docker

# relative
from ...abstract_node import AbstractNode
from ...custom_worker.config import DockerWorkerConfig
from ...node.credentials import SyftVerifyKey
from ...types.uid import UID
from ...util.util import get_queue_address
from ..response import SyftError
from .worker_image import SyftWorkerImage
from .worker_image import SyftWorkerImageTag
from .worker_image_stash import SyftWorkerImageStash
from .worker_pool import ContainerSpawnStatus
from .worker_pool import SyftWorker
from .worker_pool import WorkerOrchestrationType
from .worker_pool import WorkerStatus


def backend_container_name() -> str:
    hostname = socket.gethostname()
    return f"{hostname}-backend-1"


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


def run_container_using_docker(
    docker_client: docker.DockerClient,
    image_tag: SyftWorkerImageTag,
    worker_name: str,
    worker_count: int,
    pool_name: str,
    queue_port: int,
    debug: bool = False,
) -> ContainerSpawnStatus:
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
        # If container with given name already exists then stop it
        # and recreate it.
        existing_container = get_container(
            docker_client=docker_client,
            container_name=container_name,
        )
        if existing_container:
            existing_container.stop()

        # Extract Config from backend container
        backend_host_config = extract_config_from_backend(
            worker_name=worker_name, docker_client=docker_client
        )

        # Create List of Envs to pass
        environment = backend_host_config["environment"]
        environment["CREATE_PRODUCER"] = "false"
        environment["N_CONSUMERS"] = 1
        environment["PORT"] = str(8003 + worker_count)
        environment["HTTP_PORT"] = str(88 + worker_count)
        environment["HTTPS_PORT"] = str(446 + worker_count)
        environment["CONSUMER_SERVICE_NAME"] = pool_name
        environment["SYFT_WORKER_UID"] = syft_worker_uid
        environment["DEV_MODE"] = debug
        environment["QUEUE_PORT"] = queue_port
        environment["CONTAINER_HOST"] = "docker"

        container = docker_client.containers.run(
            image_tag.full_tag,
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

        worker = SyftWorker(
            id=syft_worker_uid,
            name=worker_name,
            container_id=container.id,
            image_hash=container.image.id,
            status=status,
            worker_pool_name=pool_name,
        )
    except Exception as e:
        error_message = f"Failed to run command in container. {worker_name} {image_tag}. {e}. {sys.stderr}"
        if container:
            worker = SyftWorker(
                name=worker_name,
                container_id=container.id,
                image_hash=container.image.id,
                status=WorkerStatus.STOPPED,
                worker_pool_name=pool_name,
            )
            container.stop()

    return ContainerSpawnStatus(
        worker_name=worker_name, worker=worker, error=error_message
    )


def run_workers_in_threads(
    node: AbstractNode, pool_name: str, number: int
) -> List[ContainerSpawnStatus]:
    results = []

    for worker_count in range(1, number + 1):
        error = None
        worker_name = f"{pool_name}-{worker_count}"
        worker = SyftWorker(
            name=worker_name,
            status=WorkerStatus.RUNNING,
            worker_pool_name=pool_name,
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
) -> List[ContainerSpawnStatus]:
    image_tag = worker_image.image_tag

    results = []

    if orchestration not in [WorkerOrchestrationType.DOCKER]:
        return SyftError(message="Only Orchestration via Docker is supported.")

    with contextlib.closing(docker.from_env()) as client:
        for worker_count in range(1, number + 1):
            worker_name = f"{pool_name}-{worker_count}"
            spawn_result = run_container_using_docker(
                docker_client=client,
                worker_name=worker_name,
                worker_count=worker_count,
                image_tag=image_tag,
                pool_name=pool_name,
                queue_port=queue_port,
                debug=dev_mode,
            )
            results.append(spawn_result)

    return results


def create_default_image(
    credentials: SyftVerifyKey,
    image_stash: SyftWorkerImageStash,
    dev_mode: bool,
    syft_version_tag: str,
):
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
