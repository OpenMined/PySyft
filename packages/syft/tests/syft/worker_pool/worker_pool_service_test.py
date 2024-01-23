# third party
from faker import Faker

# syft absolute
from syft.custom_worker.config import DockerWorkerConfig
from syft.node.worker import Worker
from syft.service.request.request import CreateCustomWorkerPoolChange
from syft.service.response import SyftSuccess
from syft.service.worker.worker_image import SyftWorkerImage
from syft.service.worker.worker_pool import WorkerPool

# relative
from ..request.request_code_accept_deny_test import get_ds_client


def test_create_image_and_pool_request_accept(faker: Faker, worker: Worker):
    """
    Test the functionality of `SyftWorkerPoolService.create_image_and_pool_request`
    when the request is accepted
    """
    # construct a root client and data scientist client for a domain
    root_client = worker.root_client
    ds_client = get_ds_client(faker, root_client, worker.guest_client)
    assert root_client.credentials != ds_client.credentials

    # the DS makes a request to create an image and a pool based on the image
    custom_dockerfile = """
        FROM openmined/grid-backend:0.8.4-beta.12

        RUN pip install recordlinkage
    """
    docker_config = DockerWorkerConfig(dockerfile=custom_dockerfile)
    docker_tag = "openmined/custom-worker-recordlinkage:latest"
    request = ds_client.api.services.worker_pool.create_image_and_pool_request(
        pool_name="recordlinkage-pool",
        num_workers=2,
        tag=docker_tag,
        config=docker_config,
        reason="I want to do some more cool data science with PySyft and Recordlinkage",
    )
    assert len(request.changes) == 2
    assert request.changes[0].config == docker_config
    assert request.changes[1].num_workers == 2
    assert request.changes[1].pool_name == "recordlinkage-pool"

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
    launched_pool = root_client.worker_pools["recordlinkage-pool"]
    assert isinstance(launched_pool, WorkerPool)
    assert len(launched_pool.worker_list) == 2


def test_create_pool_request_accept(faker: Faker, worker: Worker):
    """
    Test the functionality of `SyftWorkerPoolService.create_pool_request`
    when the request is accepted
    """
    # construct a root client and data scientist client for a domain
    root_client = worker.root_client
    ds_client = get_ds_client(faker, root_client, worker.guest_client)
    assert root_client.credentials != ds_client.credentials

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
    request = ds_client.api.services.worker_pool.pool_creation_request(
        pool_name="opendp-pool", num_workers=3, image_uid=worker_image.id
    )
    assert len(request.changes) == 1
    change = request.changes[0]
    assert isinstance(change, CreateCustomWorkerPoolChange)
    assert change.num_workers == 3
    assert change.pool_name == "opendp-pool"
    assert change.image_uid == worker_image.id

    # the root client approves the request, and the worker pool should be launched
    req_result = root_client.requests[-1].approve()
    assert isinstance(req_result, SyftSuccess)
    launched_pool = root_client.worker_pools["opendp-pool"]
    assert isinstance(launched_pool, WorkerPool)
    assert len(launched_pool.worker_list) == 3
