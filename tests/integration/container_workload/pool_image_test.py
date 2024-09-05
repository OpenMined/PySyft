# stdlib
import os
from uuid import uuid4

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.client.datasite_client import DatasiteClient
from syft.custom_worker.config import DockerWorkerConfig
from syft.custom_worker.config import PrebuiltWorkerConfig
from syft.service.request.request import Request
from syft.service.response import SyftSuccess
from syft.service.worker.worker_image import SyftWorkerImage
from syft.service.worker.worker_pool import SyftWorker
from syft.service.worker.worker_pool import WorkerPool
from syft.types.uid import UID
from syft.util.util import get_latest_tag

registry = os.getenv("SYFT_BASE_IMAGE_REGISTRY", "docker.io")
repo = "openmined/syft-backend"

if "k3d" in registry:
    tag = get_latest_tag(registry, repo)
else:
    tag = sy.__version__

external_registry = os.getenv("EXTERNAL_REGISTRY", registry)
external_registry_username = os.getenv("EXTERNAL_REGISTRY_USERNAME", None)
external_registry_password = os.getenv("EXTERNAL_REGISTRY_PASSWORD", None)


@pytest.fixture
def external_registry_uid(datasite_1_port: int) -> UID:
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )
    image_registry_list = datasite_client.api.services.image_registry.get_all()
    if len(image_registry_list) > 1:
        raise Exception("Only one registry should be present for testing")

    elif len(image_registry_list) == 1:
        assert (
            image_registry_list[0].url == external_registry
        ), "External registry different from the one set in the environment variable"
        return image_registry_list[0].id
    else:
        registry_add_result = datasite_client.api.services.image_registry.add(
            external_registry
        )

        assert isinstance(registry_add_result, sy.SyftSuccess), str(registry_add_result)

        image_registry_list = datasite_client.api.services.image_registry.get_all()
        return image_registry_list[0].id


def make_docker_config_test_case(pkg: str) -> tuple[str, str]:
    return (
        DockerWorkerConfig(
            dockerfile=(f"FROM {registry}/{repo}:{tag}\nRUN pip install {pkg}\n")
        ),
        f"openmined/custom-worker-{pkg}:latest",
    )


@pytest.mark.container_workload
def test_image_build(datasite_1_port: int, external_registry_uid: UID) -> None:
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    docker_config, docker_tag = make_docker_config_test_case("recordlinkage")

    submit_result = datasite_client.api.services.worker_image.submit(
        worker_config=docker_config
    )
    assert isinstance(submit_result, SyftSuccess)
    assert len(datasite_client.images.get_all()) == 2

    # Validate if we can get the worker image object from its config
    workerimage = datasite_client.api.services.worker_image.get_by_config(docker_config)
    # Build docker image
    docker_build_result = datasite_client.api.services.worker_image.build(
        image_uid=workerimage.id,
        tag=docker_tag,
        registry_uid=external_registry_uid,
    )
    assert isinstance(docker_build_result, SyftSuccess)

    # Refresh the worker image object
    workerimage = datasite_client.images.get_by_uid(workerimage.id)
    assert not isinstance(workerimage, sy.SyftSuccess)

    assert workerimage.is_built
    assert workerimage.image_identifier is not None
    assert workerimage.image_identifier.repo_with_tag == docker_tag
    assert workerimage.image_hash is not None


@pytest.mark.container_workload
# @pytest.mark.parametrize("prebuilt", [True, False])
@pytest.mark.parametrize("prebuilt", [False])
def test_pool_launch(
    datasite_1_port: int, external_registry_uid: UID, prebuilt: bool
) -> None:
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )

    # Submit Worker Image
    # nginx is intended to cause the startupProbe and livenessProbe to fail
    worker_config, docker_tag = (
        (PrebuiltWorkerConfig(tag="docker.io/library/nginx:latest"), None)
        if prebuilt
        else make_docker_config_test_case("opendp")
    )
    submit_result = datasite_client.api.services.worker_image.submit(
        worker_config=worker_config
    )
    assert isinstance(submit_result, SyftSuccess)

    worker_image = datasite_client.api.services.worker_image.get_by_config(
        worker_config
    )
    assert worker_image is not None

    if not worker_image.is_prebuilt:
        assert not worker_image.is_built

        # Build docker image
        docker_build_result = datasite_client.api.services.worker_image.build(
            image_uid=worker_image.id,
            tag=docker_tag,
            registry_uid=external_registry_uid,
        )
        assert isinstance(docker_build_result, SyftSuccess)

        # Push Image to External registry
        push_result = datasite_client.api.services.worker_image.push(
            worker_image.id,
            username=external_registry_username,
            password=external_registry_password,
        )
        assert isinstance(push_result, sy.SyftSuccess), str(push_result)

    # Launch a worker pool
    worker_pool_name = f"custom-worker-pool-opendp{'-prebuilt' if prebuilt else ''}"
    worker_pool_res = datasite_client.api.services.worker_pool.launch(
        pool_name=worker_pool_name,
        image_uid=worker_image.id,
        num_workers=2,
    )

    # TODO: we need to refactor this because the test is broken
    if prebuilt:
        # if the container has no liveness probe like nginx then _create_stateful_set
        # will timeout with CREATE_POOL_TIMEOUT_SEC
        # however this is currently longer than the blocking api call so we just see
        # assert "timeout" in str(worker_pool_res).lower()
        # if we lower the timout we get an exception here
        # assert "Failed to start workers" in str(worker_pool_res)
        pass
    else:
        assert all(worker.error is None for worker in worker_pool_res)

        worker_pool = datasite_client.worker_pools[worker_pool_name]
        assert len(worker_pool.worker_list) == 2

        workers = worker_pool.workers
        assert len(workers) == 2

        for worker in workers:
            assert worker.worker_pool_name == worker_pool_name
            assert worker.image.id == worker_image.id

        assert len(worker_pool.healthy_workers) == 2

        # Grab the first worker
        first_worker = workers[0]

        # Check worker Logs
        _ = datasite_client.api.services.worker.logs(uid=first_worker.id)

        # Check for worker status
        status_res = datasite_client.api.services.worker.status(uid=first_worker.id)
        assert isinstance(status_res, tuple)

        # Delete the pool's workers
        for worker in worker_pool.workers:
            res = datasite_client.api.services.worker.delete(uid=worker.id, force=True)
            assert isinstance(res, sy.SyftSuccess)

    # delete the launched pool
    res = datasite_client.api.services.worker_pool.delete(pool_name=worker_pool_name)
    assert isinstance(res, SyftSuccess), res.message


@pytest.mark.container_workload
@pytest.mark.parametrize("prebuilt", [True, False])
def test_pool_image_creation_job_requests(
    datasite_1_port: int, external_registry_uid: UID, prebuilt: bool
) -> None:
    """
    Test register ds client, ds requests to create an image and pool creation,
    do approves, then ds creates a function attached to the worker pool, then creates another
    request. DO approves and runs the function
    """
    # construct a root client and data scientist client for the test datasite
    datasite_client: DatasiteClient = sy.login(
        port=datasite_1_port, email="info@openmined.org", password="changethis"
    )
    ds_username = uuid4().hex[:8]
    ds_email = ds_username + "@example.com"
    res = datasite_client.register(
        name=ds_username,
        email=ds_email,
        password="secret_pw",
        password_verify="secret_pw",
    )
    assert isinstance(res, SyftSuccess)

    # Grant user permission to request code execution
    ds = next(u for u in datasite_client.users if u.email == ds_email)
    ds.allow_mock_execution()

    ds_client = sy.login(email=ds_email, password="secret_pw", port=datasite_1_port)

    # the DS makes a request to create an image and a pool based on the image
    worker_config, docker_tag = (
        (PrebuiltWorkerConfig(tag=f"{registry}/{repo}:{tag}"), None)
        if prebuilt
        else make_docker_config_test_case("numpy")
    )

    worker_pool_name = f"custom-worker-pool-numpy{'-prebuilt' if prebuilt else ''}"

    kwargs = {
        "pool_name": worker_pool_name,
        "num_workers": 1,
        "config": worker_config,
        "reason": "I want to do some more cool data science with PySyft",
    }
    if not prebuilt:
        kwargs.update({"tag": docker_tag, "registry_uid": external_registry_uid})

    request = ds_client.api.services.worker_pool.create_image_and_pool_request(**kwargs)
    assert isinstance(request, Request)
    assert len(request.changes) == 2
    assert request.changes[0].config == worker_config
    assert request.changes[1].num_workers == 1
    assert request.changes[1].pool_name == worker_pool_name

    # the datasite client approve the request, so the image should be built
    # and the worker pool should be launched
    for r in datasite_client.requests:
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
    data_pointer = data_action_obj.send(ds_client)

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
    for r in datasite_client.requests:
        if r.id == code_request.id:
            code_req_result = r.approve(approve_nested=True)
            break
    assert isinstance(code_req_result, SyftSuccess)

    job = ds_client.code.custom_worker_func(x=data_pointer, blocking=False)
    assert job.status.value == "created"
    job.wait()
    assert job.status.value == "completed"

    job = datasite_client.jobs.get_by_user_code_id(job.user_code_id)[-1]
    assert job.job_worker_id == worker.id

    # Validate the result received from the syft function
    result = job.wait().get()
    result_matches = result["y"] == data + 1
    assert result_matches.all()

    # Delete the workers of the launched pools
    for worker in launched_pool.workers:
        res = datasite_client.api.services.worker.delete(uid=worker.id, force=True)
        assert isinstance(res, sy.SyftSuccess)

    # delete the launched pool
    res = datasite_client.api.services.worker_pool.delete(pool_id=launched_pool.id)
    assert isinstance(res, SyftSuccess), res.message
