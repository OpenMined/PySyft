# stdlib
from textwrap import dedent
from time import sleep

# third party
from faker import Faker
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.client.domain_client import DomainClient
from syft.custom_worker.config import DockerWorkerConfig
from syft.service.request.request import Request
from syft.service.response import SyftSuccess
from syft.service.worker.worker_image import SyftWorkerImage
from syft.service.worker.worker_pool import SyftWorker
from syft.service.worker.worker_pool import WorkerPool


@pytest.mark.container_workload
def test_image_build(domain_1_port) -> None:
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # Submit Docker Worker Config
    docker_config_rl = """
        FROM openmined/grid-backend:0.8.5-beta.10
        RUN pip install recordlinkage
    """
    docker_config = DockerWorkerConfig(dockerfile=docker_config_rl)

    # Submit Worker Image
    submit_result = domain_client.api.services.worker_image.submit_dockerfile(
        docker_config=docker_config
    )
    assert isinstance(submit_result, SyftSuccess)
    assert len(domain_client.images.get_all()) == 2

    # Validate if we can get the worker image object from its config
    workerimage = domain_client.api.services.worker_image.get_by_config(docker_config)
    assert not isinstance(workerimage, sy.SyftError)

    # Build docker image
    tag_version = sy.UID().short()
    docker_tag = f"openmined/custom-worker-rl:{tag_version}"
    docker_build_result = domain_client.api.services.worker_image.build(
        image_uid=workerimage.id,
        tag=docker_tag,
    )
    assert isinstance(docker_build_result, SyftSuccess)

    # Refresh the worker image object
    workerimage = domain_client.images.get_by_uid(workerimage.id)
    assert not isinstance(workerimage, sy.SyftSuccess)

    assert workerimage.is_built
    assert workerimage.image_identifier is not None
    assert workerimage.image_identifier.repo_with_tag == docker_tag
    assert workerimage.image_hash is not None

    # Delete image
    delete_result = domain_client.api.services.worker_image.remove(uid=workerimage.id)
    assert isinstance(delete_result, sy.SyftSuccess)

    # Validate the image is successfully deleted
    assert len(domain_client.images.get_all()) == 1
    workerimage = domain_client.images.get_all()[0]
    assert workerimage.config != docker_config


@pytest.mark.container_workload
def test_pool_launch(domain_1_port) -> None:
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )
    # assert len(domain_client.worker_pools.get_all()) == 1

    # Submit Docker Worker Config
    docker_config_opendp = """
        FROM openmined/grid-backend:0.8.5-beta.10
        RUN pip install opendp
    """
    docker_config = DockerWorkerConfig(dockerfile=docker_config_opendp)

    # Submit Worker Image
    submit_result = domain_client.api.services.worker_image.submit_dockerfile(
        docker_config=docker_config
    )
    assert isinstance(submit_result, SyftSuccess)

    worker_image = domain_client.api.services.worker_image.get_by_config(docker_config)
    assert not isinstance(worker_image, sy.SyftError)
    assert worker_image is not None
    assert not worker_image.is_built

    # Build docker image
    tag_version = sy.UID().short()
    docker_tag = f"openmined/custom-worker-opendp:{tag_version}"
    docker_build_result = domain_client.api.services.worker_image.build(
        image_uid=worker_image.id,
        tag=docker_tag,
    )
    assert isinstance(docker_build_result, SyftSuccess)

    # Launch a worker pool
    pool_version = sy.UID().short()
    worker_pool_name = f"custom_worker_pool_ver{pool_version}"
    worker_pool_res = domain_client.api.services.worker_pool.launch(
        name=worker_pool_name,
        image_uid=worker_image.id,
        num_workers=3,
    )
    assert len(worker_pool_res) == 3

    assert all(worker.error is None for worker in worker_pool_res)
    # assert len(domain_client.worker_pools.get_all()) == 2

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

    # Clean the build images
    sleep(10)
    delete_result = domain_client.api.services.worker_image.remove(uid=worker_image.id)
    assert isinstance(delete_result, sy.SyftSuccess)


@pytest.mark.container_workload
def test_pool_image_creation_job_requests(domain_1_port) -> None:
    """
    Test register ds client, ds requests to create an image and pool creation,
    do approves, then ds creates a function attached to the worker pool, then creates another
    request. DO approves and runs the function
    """
    # construct a root client and data scientist client for the test domain
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )
    fake = Faker()
    ds_username = fake.user_name()
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
    docker_config_np = """
        FROM openmined/grid-backend:0.8.5-beta.10
        RUN pip install numpy
    """
    docker_config = DockerWorkerConfig(dockerfile=docker_config_np)
    tag_version = sy.UID().short()
    docker_tag = f"openmined/custom-worker-np:{tag_version}"
    pool_version = sy.UID().short()
    worker_pool_name = f"custom_worker_pool_ver{pool_version}"
    request = ds_client.api.services.worker_pool.create_image_and_pool_request(
        pool_name=worker_pool_name,
        num_workers=1,
        tag=docker_tag,
        config=docker_config,
        reason="I want to do some more cool data science with PySyft and Recordlinkage",
    )
    assert isinstance(request, Request)
    assert len(request.changes) == 2
    assert request.changes[0].config == docker_config
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
    assert worker.status.value == "Pending"
    assert worker.healthcheck.value == "✅"
    # assert worker.consumer_state.value == "Idle"
    assert isinstance(worker.logs, str)
    assert worker.job_id is None

    built_image = ds_client.api.services.worker_image.get_by_config(docker_config)
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

    custom_worker_func.code = dedent(custom_worker_func.code)
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

    # Clean the build images
    sleep(10)
    delete_result = domain_client.api.services.worker_image.remove(uid=built_image.id)
    assert isinstance(delete_result, sy.SyftSuccess)
