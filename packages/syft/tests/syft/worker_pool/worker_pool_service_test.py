# stdlib
import random
import socket
from textwrap import dedent
import time
from typing import Tuple

# third party
from faker import Faker
from hagrid.orchestra import NodeHandle
import pytest

# syft absolute
import syft as sy
from syft import ActionObject
from syft.client.domain_client import DomainClient
from syft.custom_worker.config import DockerWorkerConfig
from syft.service.request.request import CreateCustomWorkerPoolChange
from syft.service.request.request import Request
from syft.service.response import SyftSuccess
from syft.service.worker.worker_image import SyftWorkerImage
from syft.service.worker.worker_pool import WorkerPool

ADMIN_EMAIL = "info@openmined.org"
ADMIN_PASSWORD = "changethis"


def find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


@pytest.fixture
def cw_node_high() -> NodeHandle:
    """
    Fixture for an in-memory high-side node
    """
    node_port: int = find_available_port()
    queue_port: int = find_available_port()
    while queue_port == node_port:
        queue_port = find_available_port()
    random.seed()
    name = f"cw-test-domain-{random.randint(0,1000)}"
    _node: NodeHandle = sy.orchestra.launch(
        name=name,
        dev_mode=True,
        reset=True,
        # port=node_port,
        # node_side_type="high",
        create_producer=True,
        queue_port=queue_port,
        n_consumers=1,
        in_memory_workers=True,
    )
    # start the node here
    yield _node
    # clean up the node when the tests that use this fixture are done
    _node.land()


def get_clients(faker: Faker, node: NodeHandle) -> Tuple[DomainClient, DomainClient]:
    root_client: DomainClient = node.login(email=ADMIN_EMAIL, password=ADMIN_PASSWORD)
    # ds_client: DomainClient = get_ds_client(faker, root_client, worker.guest_client)
    ds_email = faker.email()
    root_client.register(
        name=faker.name(),
        email=ds_email,
        password="password",
        password_verify="password",
    )
    ds_client: DomainClient = node.login(ds_email, password="password")
    assert root_client.credentials != ds_client.credentials
    return root_client, ds_client


def test_create_image_and_pool_request_accept(
    faker: Faker, cw_node_high: NodeHandle
) -> None:
    """
    Test the functionality of `SyftWorkerPoolService.create_image_and_pool_request`
    when the request is accepted
    Note: This test only tests the worker pool with in-memory workers
    """
    # construct a root client and data scientist client for the launched node
    root_client, ds_client = get_clients(faker, cw_node_high)

    # the DS makes a request to create an image and a pool based on the image
    custom_dockerfile = """
        FROM openmined/grid-backend:0.8.4-beta.12

        RUN pip install recordlinkage
    """
    docker_config = DockerWorkerConfig(dockerfile=custom_dockerfile)
    docker_tag = "openmined/custom-worker-recordlinkage:latest"
    pool_name: str = "recordlinkage-pool"
    request = ds_client.api.services.worker_pool.create_image_and_pool_request(
        pool_name=pool_name,
        num_workers=2,
        tag=docker_tag,
        config=docker_config,
        reason="I want to do some more cool data science with PySyft and Recordlinkage",
    )
    assert len(request.changes) == 2
    assert request.changes[0].config == docker_config
    assert request.changes[1].num_workers == 2
    assert request.changes[1].pool_name == pool_name

    # the root client approve the request, so the image should be built
    # and the worker pool should be launched
    req_result = root_client.requests[-1].approve()
    assert isinstance(req_result, SyftSuccess)
    assert root_client.requests[-1].status.value == 2

    all_image_tags = [
        im.image_identifier.repo_with_tag
        for im in root_client.images.get_all()
        if im.image_identifier
    ]
    assert docker_tag in all_image_tags
    launched_pool = root_client.worker_pools[pool_name]
    assert isinstance(launched_pool, WorkerPool)
    assert len(launched_pool.worker_list) == 2

    # Delete the built image and launched worker pool
    result = root_client.api.services.worker_pool.delete(pool_name=pool_name)
    assert isinstance(result.success, str)
    assert len(root_client.worker_pools.get_all()) == 1
    assert pool_name not in [pool.name for pool in root_client.worker_pools]

    worker_image = root_client.api.services.worker_image.get_by_config(docker_config)
    result = root_client.api.services.worker_image.remove(
        uid=worker_image.id,
    )
    assert isinstance(result, SyftSuccess)


def test_create_pool_request_accept(faker: Faker, cw_node_high: NodeHandle) -> None:
    """
    Test the functionality of `SyftWorkerPoolService.create_pool_request`
    based on an already-built image when the request is accepted
    Note: This test only tests the worker pool with in-memory workers
    """
    # construct a root client and data scientist client for the launched node
    root_client, ds_client = get_clients(faker, cw_node_high)

    # the DO submits the docker config to build an image
    custom_dockerfile_str = """
        FROM openmined/grid-backend:0.8.4-beta.12

        RUN pip install opendp
    """
    docker_config = DockerWorkerConfig(dockerfile=custom_dockerfile_str)
    submit_result = root_client.api.services.worker_image.submit_dockerfile(
        docker_config=docker_config
    )
    assert isinstance(submit_result, SyftSuccess)
    assert len(root_client.images.get_all()) == 2

    # The root client builds the image
    worker_image: SyftWorkerImage = root_client.images[1]
    docker_tag = "openmined/custom-worker-opendp:latest"
    docker_build_result = root_client.api.services.worker_image.build(
        image_uid=worker_image.id,
        tag=docker_tag,
    )
    # update the worker image variable after the image was built
    worker_image: SyftWorkerImage = root_client.images[1]
    assert isinstance(docker_build_result, SyftSuccess)
    assert worker_image.image_identifier.repo_with_tag == docker_tag

    # The DS client submits a request to create a pool from an existing image
    pool_name = "opendp-pool"
    request = ds_client.api.services.worker_pool.pool_creation_request(
        pool_name=pool_name, num_workers=3, image_uid=worker_image.id
    )
    assert len(request.changes) == 1
    change = request.changes[0]
    assert isinstance(change, CreateCustomWorkerPoolChange)
    assert change.num_workers == 3
    assert change.pool_name == pool_name
    assert change.image_uid == worker_image.id

    # the root client approves the request, and the worker pool should be launched
    req_result = root_client.requests[-1].approve()
    assert isinstance(req_result, SyftSuccess)
    launched_pool = root_client.worker_pools[pool_name]
    assert isinstance(launched_pool, WorkerPool)
    assert len(launched_pool.worker_list) == 3

    # Delete the built image and launched worker pool
    result = root_client.api.services.worker_pool.delete(pool_name=pool_name)
    assert isinstance(result.success, str)
    assert len(root_client.worker_pools.get_all()) == 1
    assert pool_name not in [pool.name for pool in root_client.worker_pools]

    result = root_client.api.services.worker_image.remove(
        uid=worker_image.id,
    )
    assert isinstance(result, SyftSuccess)


def test_worker_pool_nested_jobs(faker: Faker, cw_node_high: NodeHandle) -> None:
    """
    Test scenario: DS requests to launch a pool, then submit nested syft functions
    that will be processed using the launched pool
    Note: This test only tests the worker pool with in-memory workers
    """
    # construct a root client and data scientist client for the launched in-memory node
    root_client, ds_client = get_clients(faker, cw_node_high)

    # the DS makes a request to create an image and a pool based on the image
    custom_dockerfile = """
        FROM openmined/grid-backend:0.8.4-beta.12

        RUN pip install numpy
    """
    docker_config = DockerWorkerConfig(dockerfile=custom_dockerfile)
    docker_tag: str = "openmined/custom-worker-recordlinkage:latest"
    pool_name = "syft-numpy-pool"
    request = ds_client.api.services.worker_pool.create_image_and_pool_request(
        pool_name=pool_name,
        num_workers=1,
        tag=docker_tag,
        config=docker_config,
        reason="I want to do some more cool data science with PySyft and Recordlinkage",
    )
    assert isinstance(request, Request)

    # the root client approves the pool creation request
    req_result = root_client.requests[-1].approve()
    assert isinstance(req_result, SyftSuccess)

    worker_pool: WorkerPool = root_client.worker_pools[pool_name]
    for worker in worker_pool.workers:
        assert worker.job_id is None
        # assert worker.consumer_state.value == "Detached"  # should be idle?

    # Private data of the root client
    x = ActionObject.from_obj([1, 2])
    x_ptr = x.send(root_client)

    # The DS client prepares the syft functions
    @sy.syft_function(worker_pool_name=pool_name)
    def process_batch(batch):
        print(f"starting batch {batch}")
        # stdlib
        import time

        duration = 10  # Total duration in seconds
        for i in range(duration, 0, -1):
            print(f"Countdown: {i} seconds remaining")
            time.sleep(1)  # Sleep for one second
        return batch + 1

    @sy.syft_function(worker_pool_name=pool_name)
    def aggregate_job(job_results):
        return sum(job_results)

    @sy.syft_function_single_use(x=x_ptr, worker_pool_name=pool_name)
    def process_all(domain, x):
        print("Doing processing all!")
        job_results = []
        for elem in x:
            batch_job = domain.launch_job(process_batch, batch=elem)
            job_results += [batch_job.result]
        job = domain.launch_job(aggregate_job, job_results=job_results)
        return job.result

    # The DS submits the syft functions
    aggregate_job.code = dedent(aggregate_job.code)
    res = ds_client.code.submit(aggregate_job)
    assert isinstance(res, SyftSuccess)

    process_batch.code = dedent(process_batch.code)
    res = ds_client.code.submit(process_batch)
    assert isinstance(res, SyftSuccess)

    process_all.code = dedent(process_all.code)
    code_exec_request = ds_client.code.request_code_execution(process_all)
    assert isinstance(code_exec_request, Request)

    assert (
        aggregate_job.worker_pool_name
        == process_batch.worker_pool_name
        == process_all.worker_pool_name
        == pool_name
    )

    # The root aproves the code execution request
    for req in root_client.requests:
        if req.id == code_exec_request.id:
            res = req.approve(approve_nested=True)
            assert isinstance(res, SyftSuccess)
            break

    # The DS runs the syft functions
    job = ds_client.code.process_all(x=x_ptr, blocking=False)
    print(f"on doing job with id {job.id} on the worker {job.job_worker_id}")

    # Delete the worker pool while the job is running on one of the workers
    time.sleep(2)
    result = root_client.api.services.worker_pool.delete(pool_name=pool_name)
    assert isinstance(result.success, str)
    assert len(root_client.worker_pools.get_all()) == 1
    assert pool_name not in [pool.name for pool in root_client.worker_pools]

    # Delete the built image
    worker_image = root_client.api.services.worker_image.get_by_config(docker_config)
    result2 = root_client.api.services.worker_image.remove(
        uid=worker_image.id,
    )
    assert isinstance(result2, SyftSuccess)
