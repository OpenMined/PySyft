# stdlib
import contextlib
import os
from pathlib import Path
import socket
import socketserver
import sys
from typing import Any

# third party
import docker
from docker.models.containers import Container
from kr8s.objects import Pod

# relative
from ...abstract_node import AbstractNode
from ...custom_worker.builder import CustomWorkerBuilder
from ...custom_worker.builder_types import ImageBuildResult
from ...custom_worker.builder_types import ImagePushResult
from ...custom_worker.config import DockerWorkerConfig
from ...custom_worker.config import PrebuiltWorkerConfig
from ...custom_worker.k8s import KubeUtils
from ...custom_worker.k8s import PodStatus
from ...custom_worker.runner_k8s import KubernetesRunner
from ...node.credentials import SyftVerifyKey
from ...types.uid import UID
from ...util.util import get_queue_address
from ..response import SyftError
from .image_identifier import SyftWorkerImageIdentifier
from .worker_image import SyftWorkerImage
from .worker_image_stash import SyftWorkerImageStash
from .worker_pool import ContainerSpawnStatus
from .worker_pool import SyftWorker
from .worker_pool import WorkerHealth
from .worker_pool import WorkerOrchestrationType
from .worker_pool import WorkerStatus

DEFAULT_WORKER_IMAGE_TAG = "openmined/default-worker-image-cpu:0.0.1"
DEFAULT_WORKER_POOL_NAME = "default-pool"
K8S_NODE_CREDS_NAME = "node-creds"


def backend_container_name() -> str:
    hostname = socket.gethostname()
    service_name = os.getenv("SERVICE", "backend")
    return f"{hostname}-{service_name}-1"


def get_container(
    docker_client: docker.DockerClient, container_name: str
) -> Container | None:
    try:
        existing_container = docker_client.containers.get(container_name)
    except docker.errors.NotFound:
        existing_container = None

    return existing_container


def extract_config_from_backend(
    worker_name: str, docker_client: docker.DockerClient
) -> dict[str, Any]:
    # Existing main backend container
    backend_container = get_container(
        docker_client, container_name=backend_container_name()
    )

    # Config with defaults
    extracted_config: dict[str, Any] = {
        "volume_binds": {},
        "network_mode": None,
        "environment": {},
    }

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


def get_free_tcp_port() -> int:
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
    username: str | None = None,
    password: str | None = None,
    registry_url: str | None = None,
) -> ContainerSpawnStatus:
    if not worker_image.is_built:
        raise ValueError("Image must be built before running it.")

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
        environment["CONSUMER_SERVICE_NAME"] = pool_name
        environment["SYFT_WORKER_UID"] = syft_worker_uid
        environment["DEV_MODE"] = debug
        environment["QUEUE_PORT"] = queue_port
        environment["CONTAINER_HOST"] = "docker"

        if worker_image.image_identifier is None:
            raise ValueError(f"Image {worker_image} does not have an identifier")

        container = docker_client.containers.run(
            image=worker_image.image_identifier.full_name_with_tag,
            name=f"{hostname}-{worker_name}",
            detach=True,
            # auto_remove=True,
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
) -> list[ContainerSpawnStatus]:
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
                "Failed to start consumer for "
                f"pool={pool_name} worker={worker_name}. Error: {e}"
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


def prepare_kubernetes_pool_env(
    runner: KubernetesRunner, env_vars: dict
) -> tuple[list, dict]:
    # get current backend pod name
    backend_pod_name = os.getenv("K8S_POD_NAME")
    if not backend_pod_name:
        raise ValueError("Pod name not provided in environment variable")

    # get current backend's credentials path
    creds_path: str | Path | None = os.getenv("CREDENTIALS_PATH")
    if not creds_path:
        raise ValueError("Credentials path not provided")

    creds_path = Path(creds_path)
    if creds_path is not None and not creds_path.exists():
        raise ValueError("Credentials file does not exist")

    # create a secret for the node credentials owned by the backend, not the pool.
    node_secret = KubeUtils.create_secret(
        secret_name=K8S_NODE_CREDS_NAME,
        type="Opaque",
        component=backend_pod_name,
        data={creds_path.name: creds_path.read_text()},
        encoded=False,
    )

    # clone and patch backend environment variables
    backend_env = runner.get_pod_env_vars(backend_pod_name) or []
    env_vars_: list = KubeUtils.patch_env_vars(backend_env, env_vars)
    mount_secrets = {
        node_secret.metadata.name: {
            "mountPath": str(creds_path),
            "subPath": creds_path.name,
        },
    }

    return env_vars_, mount_secrets


def create_kubernetes_pool(
    runner: KubernetesRunner,
    tag: str,
    pool_name: str,
    replicas: int,
    queue_port: int,
    debug: bool,
    reg_username: str | None = None,
    reg_password: str | None = None,
    reg_url: str | None = None,
    **kwargs: Any,
) -> list[Pod] | SyftError:
    pool = None
    error = False

    try:
        print(
            "Creating new pool "
            f"name={pool_name} "
            f"tag={tag} "
            f"replicas={replicas}"
        )

        env_vars, mount_secrets = prepare_kubernetes_pool_env(
            runner,
            {
                "SYFT_WORKER": "True",
                "DEV_MODE": f"{debug}",
                "QUEUE_PORT": f"{queue_port}",
                "CONSUMER_SERVICE_NAME": pool_name,
                "N_CONSUMERS": "1",
                "CREATE_PRODUCER": "False",
                "INMEMORY_WORKERS": "False",
            },
        )

        # run the pool with args + secret
        pool = runner.create_pool(
            pool_name=pool_name,
            tag=tag,
            replicas=replicas,
            env_vars=env_vars,
            mount_secrets=mount_secrets,
            reg_username=reg_username,
            reg_password=reg_password,
            reg_url=reg_url,
        )
    except Exception as e:
        error = True
        return SyftError(message=f"Failed to start workers {e}")
    finally:
        if error and pool:
            pool.delete()

    return runner.get_pool_pods(pool_name=pool_name)


def scale_kubernetes_pool(
    runner: KubernetesRunner,
    pool_name: str,
    replicas: int,
) -> list[Pod] | SyftError:
    pool = runner.get_pool(pool_name)
    if not pool:
        return SyftError(message=f"Pool does not exist. name={pool_name}")

    try:
        print(f"Scaling pool name={pool_name} to replicas={replicas}")
        runner.scale_pool(pool_name=pool_name, replicas=replicas)
    except Exception as e:
        return SyftError(message=f"Failed to scale workers {e}")

    return runner.get_pool_pods(pool_name=pool_name)


def run_workers_in_kubernetes(
    worker_image: SyftWorkerImage,
    worker_count: int,
    pool_name: str,
    queue_port: int,
    start_idx: int = 0,
    debug: bool = False,
    reg_username: str | None = None,
    reg_password: str | None = None,
    reg_url: str | None = None,
    **kwargs: Any,
) -> list[ContainerSpawnStatus] | SyftError:
    spawn_status = []
    runner = KubernetesRunner()

    if not runner.exists(pool_name=pool_name):
        if worker_image.image_identifier is not None:
            pool_pods = create_kubernetes_pool(
                runner=runner,
                tag=worker_image.image_identifier.full_name_with_tag,
                pool_name=pool_name,
                replicas=worker_count,
                queue_port=queue_port,
                debug=debug,
                reg_username=reg_username,
                reg_password=reg_password,
                reg_url=reg_url,
            )
        else:
            return SyftError(
                message=f"image with uid {worker_image.id} does not have an image identifier"
            )
    else:
        pool_pods = scale_kubernetes_pool(runner, pool_name, worker_count)

        if isinstance(pool_pods, list) and len(pool_pods) > 0:
            # slice only those pods that we're interested in
            pool_pods = pool_pods[start_idx:]

    if isinstance(pool_pods, SyftError):
        return pool_pods

    # create worker object
    for pod in pool_pods:
        status: PodStatus | WorkerStatus | None = runner.get_pod_status(pod)
        status, healthcheck, error = map_pod_to_worker_status(status)

        # this worker id will be the same as the one in the worker
        syft_worker_uid = UID.with_seed(pod.metadata.name)

        worker = SyftWorker(
            id=syft_worker_uid,
            name=pod.metadata.name,
            container_id=None,
            status=status,
            healthcheck=healthcheck,
            image=worker_image,
            worker_pool_name=pool_name,
        )

        spawn_status.append(
            ContainerSpawnStatus(
                worker_name=pod.metadata.name,
                worker=worker,
                error=error,
            )
        )

    return spawn_status


def map_pod_to_worker_status(
    status: PodStatus,
) -> tuple[WorkerStatus, WorkerHealth, str | None]:
    worker_status = None
    worker_healthcheck = None
    worker_error = None

    # check if pod is ready through pod.status.condition.Ready & pod.status.condition.ContainersReady
    pod_ready = status.condition.ready and status.condition.containers_ready

    if not pod_ready:
        # extract error if not ready
        worker_error = f"{status.container.reason}: {status.container.message}"

    # map readiness to status - it's either running or pending.
    # closely relates to pod.status.phase, but avoiding as it is not as detailed as pod.status.conditions
    worker_status = WorkerStatus.RUNNING if pod_ready else WorkerStatus.PENDING

    # TODO: update these values based on actual runtime probes instead of kube pod statuses
    # if there are any errors, then healthcheck is unhealthy
    worker_healthcheck = (
        WorkerHealth.UNHEALTHY if worker_error else WorkerHealth.HEALTHY
    )

    return worker_status, worker_healthcheck, worker_error


def run_containers(
    pool_name: str,
    worker_image: SyftWorkerImage,
    number: int,
    orchestration: WorkerOrchestrationType,
    queue_port: int,
    dev_mode: bool = False,
    start_idx: int = 0,
    reg_username: str | None = None,
    reg_password: str | None = None,
    reg_url: str | None = None,
) -> list[ContainerSpawnStatus] | SyftError:
    results = []

    if not worker_image.is_built:
        return SyftError(message="Image must be built before running it.")

    print(f"Starting workers with start_idx={start_idx} count={number}")

    if orchestration == WorkerOrchestrationType.DOCKER:
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
                    username=reg_username,
                    password=reg_password,
                    registry_url=reg_url,
                )
                results.append(spawn_result)
    elif orchestration == WorkerOrchestrationType.KUBERNETES:
        return run_workers_in_kubernetes(
            worker_image=worker_image,
            worker_count=number,
            pool_name=pool_name,
            queue_port=queue_port,
            debug=dev_mode,
            start_idx=start_idx,
            reg_username=reg_username,
            reg_password=reg_password,
            reg_url=reg_url,
        )

    return results


def create_default_image(
    credentials: SyftVerifyKey,
    image_stash: SyftWorkerImageStash,
    tag: str,
    in_kubernetes: bool = False,
) -> SyftError | SyftWorkerImage:
    # TODO: Hardcode worker dockerfile since not able to COPY
    # worker_cpu.dockerfile to backend in backend.dockerfile.

    # default_cpu_dockerfile = get_syft_cpu_dockerfile()
    # DockerWorkerConfig.from_path(default_cpu_dockerfile)

    if not in_kubernetes:
        default_cpu_dockerfile = f"""ARG SYFT_VERSION_TAG='{tag}' \n"""
        default_cpu_dockerfile += """FROM openmined/grid-backend:${SYFT_VERSION_TAG}
        ARG PYTHON_VERSION="3.12"
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
        _new_image = SyftWorkerImage(
            config=worker_config,
            created_by=credentials,
        )
    else:
        # in k8s we don't need to build the image, just the tag of backend is enough
        worker_config = PrebuiltWorkerConfig(
            tag=tag,
            description="Prebuilt default worker image",
        )

        # create SyftWorkerImage from a pre-built image
        _new_image = SyftWorkerImage(
            config=worker_config,
            created_by=credentials,
            image_identifier=SyftWorkerImageIdentifier.from_str(tag),
        )

    result = image_stash.get_by_docker_config(
        credentials=credentials,
        config=worker_config,
    )

    if result.ok() is None:
        result = image_stash.set(credentials, _new_image)
        if result.is_err():
            return SyftError(message=f"Failed to save image stash: {result.err()}")

    default_syft_image = result.ok()

    return default_syft_image


def _get_healthcheck_based_on_status(status: WorkerStatus) -> WorkerHealth:
    if status in [WorkerStatus.PENDING, WorkerStatus.RUNNING]:
        return WorkerHealth.HEALTHY
    else:
        return WorkerHealth.UNHEALTHY


def image_build(
    image: SyftWorkerImage, **kwargs: dict[str, Any]
) -> ImageBuildResult | SyftError:
    if image.image_identifier is not None:
        full_tag = image.image_identifier.full_name_with_tag
        try:
            builder = CustomWorkerBuilder()
            return builder.build_image(
                config=image.config,
                tag=full_tag,
                # rm=True,
                # forcerm=True,
                **kwargs,
            )
        except docker.errors.APIError as e:
            return SyftError(
                message=f"Docker API error when building '{full_tag}'. Reason - {e}"
            )
        except docker.errors.DockerException as e:
            return SyftError(
                message=f"Docker exception when building '{full_tag}'. Reason - {e}"
            )
        except Exception as e:
            return SyftError(
                message=f"Unknown exception when building '{full_tag}'. Reason - {e}"
            )
    else:
        return SyftError(
            message=f"image with uid {image.id} does not have an image identifier"
        )


def image_push(
    image: SyftWorkerImage,
    username: str | None = None,
    password: str | None = None,
) -> ImagePushResult | SyftError:
    if image.image_identifier is not None:
        full_tag = image.image_identifier.full_name_with_tag
        try:
            builder = CustomWorkerBuilder()
            result = builder.push_image(
                # this should be consistent with docker build command
                tag=image.image_identifier.full_name_with_tag,
                registry_url=image.image_identifier.registry_host,
                username=username,
                password=password,
            )

            if "error" in result.logs.lower() or result.exit_code:
                return SyftError(
                    message=f"Failed to push {full_tag}. "
                    f"Exit code: {result.exit_code}. "
                    f"Logs:\n{result.logs}"
                )

            return result
        except docker.errors.APIError as e:
            return SyftError(message=f"Docker API error when pushing {full_tag}. {e}")
        except docker.errors.DockerException as e:
            return SyftError(
                message=f"Docker exception when pushing {full_tag}. Reason - {e}"
            )
        except Exception as e:
            return SyftError(
                message=f"Unknown exception when pushing {image.image_identifier}. Reason - {e}"
            )
    else:
        return SyftError(
            message=f"image with uid {image.id} does not have an "
            "image identifier and tag, hence we can't push it."
        )


def get_orchestration_type() -> WorkerOrchestrationType:
    """Returns orchestration type from env. Defaults to Python."""

    orchstration_type_ = os.getenv("CONTAINER_HOST")
    return (
        WorkerOrchestrationType(orchstration_type_.lower())
        if orchstration_type_
        else WorkerOrchestrationType.PYTHON
    )
