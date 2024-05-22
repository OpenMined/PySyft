# stdlib
import os
from uuid import uuid4

# third party
import numpy as np
import pytest
import requests

# syft absolute
import syft as sy
from syft.client.domain_client import DomainClient
from syft.custom_worker.config import DockerWorkerConfig
from syft.custom_worker.config import PrebuiltWorkerConfig
from syft.service.request.request import Request
from syft.service.response import SyftError
from syft.service.response import SyftSuccess
from syft.service.worker.worker_image import SyftWorkerImage
from syft.service.worker.worker_pool import SyftWorker
from syft.service.worker.worker_pool import WorkerPool
from syft.types.uid import UID

registry = os.getenv("SYFT_BASE_IMAGE_REGISTRY", "docker.io")
repo = "openmined/grid-backend"

if "k3d" in registry:
    res = requests.get(url=f"http://{registry}/v2/{repo}/tags/list")
    tag = res.json()["tags"][0]
else:
    tag = sy.__version__

external_registry = os.getenv("EXTERNAL_REGISTRY", registry)
external_registry_username = os.getenv("EXTERNAL_REGISTRY_USERNAME", None)
external_registry_password = os.getenv("EXTERNAL_REGISTRY_PASSWORD", None)


@pytest.fixture
def external_registry_uid(domain_1_port: int) -> UID:
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )
    image_registry_list = domain_client.api.services.image_registry.get_all()
    if len(image_registry_list) > 1:
        raise Exception("Only one registry should be present for testing")

    elif len(image_registry_list) == 1:
        assert (
            image_registry_list[0].url == external_registry
        ), "External registry different from the one set in the environment variable"
        return image_registry_list[0].id
    else:
        registry_add_result = domain_client.api.services.image_registry.add(
            external_registry
        )

        assert isinstance(registry_add_result, sy.SyftSuccess), str(registry_add_result)

        image_registry_list = domain_client.api.services.image_registry.get_all()
        return image_registry_list[0].id


def make_docker_config_test_case(pkg: str) -> tuple[str, str]:
    return (
        DockerWorkerConfig(
            dockerfile=(f"FROM {registry}/{repo}:{tag}\nRUN pip install {pkg}\n")
        ),
        f"openmined/custom-worker-{pkg}:latest",
    )


@pytest.mark.container_workload
def test_image_build(domain_1_port: int, external_registry_uid: UID) -> None:
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    docker_config, docker_tag = make_docker_config_test_case("recordlinkage")

    submit_result = domain_client.api.services.worker_image.submit_container_image(
        worker_config=docker_config
    )
    assert isinstance(submit_result, SyftSuccess)
    assert len(domain_client.images.get_all()) == 2

    # Validate if we can get the worker image object from its config
    workerimage = domain_client.api.services.worker_image.get_by_config(docker_config)
    assert not isinstance(workerimage, sy.SyftError)

    # Build docker image
    docker_build_result = domain_client.api.services.worker_image.build(
        image_uid=workerimage.id,
        tag=docker_tag,
        registry_uid=external_registry_uid,
    )
    assert isinstance(docker_build_result, SyftSuccess)

    # Refresh the worker image object
    workerimage = domain_client.images.get_by_uid(workerimage.id)
    assert not isinstance(workerimage, sy.SyftSuccess)

    assert workerimage.is_built
    assert workerimage.image_identifier is not None
    assert workerimage.image_identifier.repo_with_tag == docker_tag
    assert workerimage.image_hash is not None


@pytest.mark.container_workload
@pytest.mark.parametrize("prebuilt", [True, False])
def test_pool_launch(
    domain_1_port: int, external_registry_uid: UID, prebuilt: bool
) -> None:
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # Submit Worker Image
    worker_config, docker_tag = (
        (PrebuiltWorkerConfig(tag=(_tag := "docker.io/library/nginx:latest")), _tag)
        if prebuilt
        else make_docker_config_test_case("opendp")
    )
    submit_result = domain_client.api.services.worker_image.submit_container_image(
        worker_config=worker_config
    )
    assert isinstance(submit_result, SyftSuccess)

    worker_image = domain_client.api.services.worker_image.get_by_config(worker_config)
    assert not isinstance(worker_image, sy.SyftError)
    assert worker_image is not None

    if not worker_image.is_prebuilt:
        assert not worker_image.is_built

        # Build docker image
        docker_build_result = domain_client.api.services.worker_image.build(
            image_uid=worker_image.id,
            tag=docker_tag,
            registry_uid=external_registry_uid,
        )
        assert isinstance(docker_build_result, SyftSuccess)

        # Push Image to External registry
        push_result = domain_client.api.services.worker_image.push(
            worker_image.id,
            username=external_registry_username,
            password=external_registry_password,
        )
        assert isinstance(push_result, sy.SyftSuccess), str(push_result)

    # Launch a worker pool
    worker_pool_name = f"custom-worker-pool-opendp{'-prebuilt' if prebuilt else ''}"
    worker_pool_res = domain_client.api.services.worker_pool.launch(
        name=worker_pool_name,
        image_uid=worker_image.id,
        num_workers=3,
    )
    assert not isinstance(worker_pool_res, SyftError)

    assert all(worker.error is None for worker in worker_pool_res)

    worker_pool = domain_client.worker_pools[worker_pool_name]
    assert len(worker_pool.worker_list) == 3

    workers = worker_pool.workers
    assert len(workers) == 3

    for worker in workers:
        assert worker.worker_pool_name == worker_pool_name
        assert worker.image.id == worker_image.id

    assert len(worker_pool.healthy_workers) == 3

    # Grab the first worker
    first_worker = workers[0]

    # Check worker Logs
    logs = domain_client.api.services.worker.logs(uid=first_worker.id)
    assert not isinstance(logs, sy.SyftError)

    # Check for worker status
    status_res = domain_client.api.services.worker.status(uid=first_worker.id)
    assert not isinstance(status_res, sy.SyftError)
    assert isinstance(status_res, tuple)

    # Delete the pool's workers
    for worker in worker_pool.workers:
        res = domain_client.api.services.worker.delete(uid=worker.id, force=True)
        assert isinstance(res, sy.SyftSuccess)

    # TODO: delete the launched pool


@pytest.mark.container_workload
@pytest.mark.parametrize("prebuilt", [True, False])
def test_pool_image_creation_job_requests(
    domain_1_port: int, external_registry_uid: UID, prebuilt: bool
) -> None:
    """
    Test register ds client, ds requests to create an image and pool creation,
    do approves, then ds creates a function attached to the worker pool, then creates another
    request. DO approves and runs the function
    """
    # construct a root client and data scientist client for the test domain
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )
    ds_username = uuid4().hex[:8]
    ds_email = ds_username + "@example.com"
    res = domain_client.register(
        name=ds_username,
        email=ds_email,
        password="secret_pw",
        password_verify="secret_pw",
    )
    assert isinstance(res, SyftSuccess)
    ds_client = sy.login(email=ds_email, password="secret_pw", port=domain_1_port)

    # the DS makes a request to create an image and a pool based on the image
    worker_config, docker_tag = (
        (
            PrebuiltWorkerConfig(tag=(_tag := f"{registry}/{repo}:{tag}")),
            _tag,
        )
        if prebuilt
        else make_docker_config_test_case("numpy")
    )

    worker_pool_name = f"custom-worker-pool-numpy{'-prebuilt' if prebuilt else ''}"
    request = ds_client.api.services.worker_pool.create_image_and_pool_request(
        pool_name=worker_pool_name,
        num_workers=1,
        tag=docker_tag,
        config=worker_config,
        reason="I want to do some more cool data science with PySyft",
        registry_uid=external_registry_uid,
    )
    assert isinstance(request, Request)
    assert len(request.changes) == 2
    assert request.changes[0].config == worker_config
    assert request.changes[1].num_workers == 1
    assert request.changes[1].pool_name == worker_pool_name

    # the domain client approve the request, so the image should be built
    # and the worker pool should be launched
    for r in domain_client.requests:
        if r.id == request.id:
            req_result = r.approve()
            break
    assert isinstance(req_result, SyftSuccess)

    launched_pool = ds_client.api.services.worker_pool.get_by_name(worker_pool_name)
    assert isinstance(launched_pool, WorkerPool)
    assert launched_pool.name == worker_pool_name
    assert len(launched_pool.worker_list) == 1

    worker: SyftWorker = launched_pool.workers[0]
    assert launched_pool.name in worker.name
    assert worker.status.value == "Running"
    assert worker.healthcheck.value == "✅"
    # assert worker.consumer_state.value == "Idle"
    assert isinstance(worker.logs, str)
    assert worker.job_id is None

    built_image = ds_client.api.services.worker_image.get_by_config(worker_config)
    assert isinstance(built_image, SyftWorkerImage)
    assert built_image.id == launched_pool.image.id
    assert worker.image.id == built_image.id

    # Dataset
    data = np.array([1, 2, 3])
    data_action_obj = sy.ActionObject.from_obj(data)
    data_pointer = domain_client.api.services.action.set(data_action_obj)

    # Function
    @sy.syft_function(
        input_policy=sy.ExactMatch(x=data_pointer),
        output_policy=sy.SingleExecutionExactOutput(),
        worker_pool_name=launched_pool.name,
    )
    def custom_worker_func(x):
        return {"y": x + 1}

    assert custom_worker_func.worker_pool_name == launched_pool.name
    # Request code execution
    code_request = ds_client.code.request_code_execution(custom_worker_func)
    assert isinstance(code_request, Request)
    assert code_request.status.value == 0  # pending
    for r in domain_client.requests:
        if r.id == code_request.id:
            code_req_result = r.approve(approve_nested=True)
            break
    assert isinstance(code_req_result, SyftSuccess)

    job = ds_client.code.custom_worker_func(x=data_pointer, blocking=False)
    assert job.status.value == "created"
    job.wait()
    assert job.status.value == "completed"

    job = domain_client.jobs[-1]
    assert job.job_worker_id == worker.id

    # Validate the result received from the syft function
    result = job.wait().get()
    result_matches = result["y"] == data + 1
    assert result_matches.all()

    # Delete the workers of the launched pools
    for worker in launched_pool.workers:
        res = domain_client.api.services.worker.delete(uid=worker.id, force=True)
        assert isinstance(res, sy.SyftSuccess)

    # TODO: delete the launched pool
