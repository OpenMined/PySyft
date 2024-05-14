# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy
from syft.custom_worker.config import DockerWorkerConfig
from syft.custom_worker.config import PrebuiltWorkerConfig
from syft.custom_worker.config import WorkerConfig
from syft.node.worker import Worker
from syft.service.request.request import CreateCustomWorkerPoolChange
from syft.service.response import SyftSuccess
from syft.service.worker.worker_image import SyftWorkerImage
from syft.service.worker.worker_pool import WorkerPool

# relative
from ..request.request_code_accept_deny_test import get_ds_client

PREBUILT_IMAGE_TAG = f"openmined/grid-backend:{sy.__version__}"

CUSTOM_DOCKERFILE = f"""
FROM {PREBUILT_IMAGE_TAG}

RUN pip install recordlinkage
"""

CUSTOM_IMAGE_TAG = "openmined/custom-worker-recordlinkage:latest"

DOCKER_CONFIG_TEST_CASES_WITH_N_IMAGES = [
    (
        CUSTOM_IMAGE_TAG,
        DockerWorkerConfig(dockerfile=CUSTOM_DOCKERFILE),
        2,  # total number of images.
        # 2 since we pull a pre-built image (1) as the base image to build a custom image (2)
    ),
    (PREBUILT_IMAGE_TAG, PrebuiltWorkerConfig(tag=PREBUILT_IMAGE_TAG), 1),
]

DOCKER_CONFIG_TEST_CASES = [
    test_case[:2] for test_case in DOCKER_CONFIG_TEST_CASES_WITH_N_IMAGES
]


@pytest.mark.parametrize("docker_tag,docker_config", DOCKER_CONFIG_TEST_CASES)
def test_create_image_and_pool_request_accept(
    faker: Faker, worker: Worker, docker_tag: str, docker_config: WorkerConfig
) -> None:
    """
    Test the functionality of `SyftWorkerPoolService.create_image_and_pool_request`
    when the request is accepted
    """
    # construct a root client and data scientist client for a domain
    root_client = worker.root_client
    ds_client = get_ds_client(faker, root_client, worker.guest_client)
    assert root_client.credentials != ds_client.credentials

    # the DS makes a request to create an image and a pool based on the image
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


@pytest.mark.parametrize(
    "docker_tag,docker_config,n_images",
    DOCKER_CONFIG_TEST_CASES_WITH_N_IMAGES,
)
def test_create_pool_request_accept(
    faker: Faker,
    worker: Worker,
    docker_tag: str,
    docker_config: WorkerConfig,
    n_images: int,
) -> None:
    """
    Test the functionality of `SyftWorkerPoolService.create_pool_request`
    when the request is accepted
    """
    # construct a root client and data scientist client for a domain
    root_client = worker.root_client
    ds_client = get_ds_client(faker, root_client, worker.guest_client)
    assert root_client.credentials != ds_client.credentials

    # the DO submits the docker config to build an image
    submit_result = root_client.api.services.worker_image.submit_container_image(
        docker_config=docker_config
    )
    assert isinstance(submit_result, SyftSuccess)
    assert len(root_client.images.get_all()) == n_images

    # The root client builds the image
    worker_image: SyftWorkerImage = root_client.images[-1]
    if not worker_image.is_prebuilt:
        docker_build_result = root_client.api.services.worker_image.build(
            image_uid=worker_image.id,
            tag=docker_tag,
        )
        # update the worker image variable after the image was built
        worker_image: SyftWorkerImage = root_client.images[-1]
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


WORKER_CONFIGS = [test_case[1] for test_case in DOCKER_CONFIG_TEST_CASES]


@pytest.mark.parametrize("docker_config", WORKER_CONFIGS)
def test_get_by_worker_config(
    worker: Worker,
    docker_config: WorkerConfig,
) -> None:
    root_client = worker.root_client
    for config in WORKER_CONFIGS:
        root_client.api.services.worker_image.submit_container_image(
            docker_config=config
        )

    worker_image = root_client.api.services.worker_image.get_by_config(docker_config)
    assert worker_image.config == docker_config
