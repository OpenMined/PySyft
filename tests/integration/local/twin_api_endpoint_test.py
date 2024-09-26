# stdlib
from collections.abc import Callable
import os
import sys
import time

# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy
from syft.client.datasite_client import DatasiteClient
from syft.service.api.api import TwinAPIEndpoint
from syft.service.response import SyftError
from syft.service.response import SyftSuccess

JOB_TIMEOUT = 20


def get_external_registry() -> str:
    """Get the external registry to use for the worker image."""
    return os.environ.get("EXTERNAL_REGISTRY", "docker.io")


def get_worker_tag() -> str:
    """Get the worker tag to use for the worker image."""
    return os.environ.get("PRE_BUILT_WORKER_TAG", f"openmined/backend:{sy.__version__}")


def public_function(
    context,
) -> str:
    return "Public Function Execution"


def private_function(
    context,
) -> str:
    return "Private Function Execution"


def get_twin_api_endpoint(worker_pool_name: str) -> TwinAPIEndpoint:
    """Get a twin API endpoint with a custom worker pool name."""

    public_func = sy.api_endpoint_method(settings={"Hello": "Public"})(public_function)
    pvt_func = sy.api_endpoint_method(settings={"Hello": "Private"})(private_function)

    new_endpoint = sy.TwinAPIEndpoint(
        path="second.query",
        mock_function=public_func,
        private_function=pvt_func,
        description="Lore ipsulum ...",
        worker_pool_name=worker_pool_name,
    )

    return new_endpoint


faker = Faker()


def get_ds_client(client: DatasiteClient) -> DatasiteClient:
    """Get a datasite client with a registered user."""
    pwd = faker.password()
    email = faker.email()
    client.register(
        name=faker.name(),
        email=email,
        password=pwd,
        password_verify=pwd,
    )
    return client.login(email=email, password=pwd)


def get_syft_function(worker_pool_name: str, endpoint: Callable) -> Callable:
    @sy.syft_function_single_use(endpoint=endpoint, worker_pool_name=worker_pool_name)
    def job_function(endpoint):
        return endpoint()

    return job_function


def submit_project(ds_client: DatasiteClient, syft_function: Callable):
    # Create a new project
    new_project = sy.Project(
        name=f"Project - {faker.text(max_nb_chars=20)}",
        description="Hi, I want to calculate the trade volume in million's with my cool code.",
        members=[ds_client],
    )

    result = new_project.create_code_request(syft_function, ds_client)
    assert isinstance(result, SyftSuccess)


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
# @pytest.mark.local_server
def test_twin_api_with_custom_worker(full_high_worker):
    high_client = full_high_worker.login(
        email="info@openmined.org", password="changethis"
    )

    worker_pool_name = "custom-worker-pool"

    external_registry = get_external_registry()
    worker_docker_tag = get_worker_tag()

    # Create pre-built worker image
    docker_config = sy.PrebuiltWorkerConfig(
        tag=f"{external_registry}/{worker_docker_tag}"
    )

    # Submit the worker image
    submit_result = high_client.api.services.worker_image.submit(
        worker_config=docker_config
    )

    # Check if the submission was successful
    assert not isinstance(submit_result, SyftError), submit_result

    # Get the worker image
    worker_image = high_client.images.get_all()[-1]

    launch_result = high_client.api.services.worker_pool.launch(
        pool_name=worker_pool_name,
        image_uid=worker_image.id,
        num_workers=2,
    )

    # Check if the worker pool was launched successfully
    assert not isinstance(launch_result, SyftError), launch_result

    # Add the twin API endpoint
    twin_api_endpoint = get_twin_api_endpoint(worker_pool_name)
    twin_endpoint_result = high_client.api.services.api.add(endpoint=twin_api_endpoint)

    # Check if the twin API endpoint was added successfully
    assert isinstance(twin_endpoint_result, SyftSuccess)

    # validate the number of endpoints
    assert len(high_client.api.services.api.api_endpoints()) == 1

    # refresh the client
    high_client.refresh()

    # Get datasite client
    high_client_ds = get_ds_client(high_client)

    # Execute the public endpoint
    mock_endpoint_result = high_client_ds.api.services.second.query()
    assert mock_endpoint_result == "Public Function Execution"

    # Get the syft function
    custom_function = get_syft_function(
        worker_pool_name, high_client_ds.api.services.second.query
    )

    # Submit the project
    submit_project(high_client_ds, custom_function)

    ds_email = high_client_ds.logged_in_user

    # Approve the request
    for r in high_client.requests.get_all():
        if r.requesting_user_email == ds_email:
            r.approve()

    private_func_result_job = high_client_ds.code.job_function(
        endpoint=high_client_ds.api.services.second.query, blocking=False
    )

    # Wait for the job to complete
    job_start_time = time.time()
    while True:
        # Check if the job is resolved
        _ = private_func_result_job.resolved

        if private_func_result_job.resolve:
            break

        # Check if the job is timed out
        if time.time() - job_start_time > JOB_TIMEOUT:
            raise TimeoutError(f"Job did not complete in given time: {JOB_TIMEOUT}")
        time.sleep(1)

    # Check if the job worker is the same as the worker pool name
    private_func_job = high_client_ds.jobs.get(private_func_result_job.id)

    assert private_func_job is not None

    # Check if job is assigned to a worker
    assert private_func_job.job_worker_id is not None

    # Check if the job worker is the same as the worker pool name
    assert private_func_job.worker.worker_pool_name == worker_pool_name

    # Check if the job was successful
    assert private_func_result_job.resolved
    private_func_result = private_func_result_job.result

    assert not isinstance(private_func_result, SyftError), private_func_result

    assert private_func_result.get() == "Private Function Execution"
