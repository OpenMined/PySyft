# third party
import pytest

# syft absolute
import syft as sy
from syft.client.client import SyftClient
from syft.custom_worker.config import DockerWorkerConfig
from syft.custom_worker.config import PrebuiltWorkerConfig
from syft.custom_worker.config import WorkerConfig
from syft.server.worker import Worker
from syft.service.request.request import CreateCustomWorkerPoolChange
from syft.service.response import SyftSuccess
from syft.service.worker.worker_image import SyftWorkerImage
from syft.service.worker.worker_pool import WorkerPool

PREBUILT_IMAGE_TAG = f"docker.io/openmined/syft-backend:{sy.__version__}"

CUSTOM_DOCKERFILE = f"""
FROM {PREBUILT_IMAGE_TAG}

RUN pip install recordlinkage
"""

CUSTOM_IMAGE_TAG = "docker.io/openmined/custom-worker-recordlinkage:latest"

WORKER_CONFIG_TEST_CASES_WITH_N_IMAGES = [
    (
        CUSTOM_IMAGE_TAG,
        DockerWorkerConfig(dockerfile=CUSTOM_DOCKERFILE),
        2,  # total number of images.
        # 2 since we pull a pre-built image (1) as the base image to build a custom image (2)
    ),
    (None, PrebuiltWorkerConfig(tag=PREBUILT_IMAGE_TAG), 2),
]

WORKER_CONFIG_TEST_CASES = [
    test_case[:2] for test_case in WORKER_CONFIG_TEST_CASES_WITH_N_IMAGES
]


@pytest.mark.parametrize("docker_tag,worker_config", WORKER_CONFIG_TEST_CASES)
def test_create_image_and_pool_request_accept(
    worker: Worker,
    docker_tag: str,
    worker_config: WorkerConfig,
    ds_client: SyftClient,
) -> None:
    """
    Test the functionality of `SyftWorkerPoolService.create_image_and_pool_request`
    when the request is accepted
    """
    # construct a root client and data scientist client for a datasite
    root_client = worker.root_client
    assert root_client.credentials != ds_client.credentials

    # the DS makes a request to create an image and a pool based on the image
    request = ds_client.api.services.worker_pool.create_image_and_pool_request(
        pool_name="recordlinkage-pool",
        num_workers=2,
        tag=docker_tag,
        config=worker_config,
        reason="I want to do some more cool data science with PySyft and Recordlinkage",
    )
    assert len(request.changes) == 2
    assert request.changes[0].config == worker_config
    assert request.changes[1].num_workers == 2
    assert request.changes[1].pool_name == "recordlinkage-pool"

    # the root client approve the request, so the image should be built
    # and the worker pool should be launched
    req_result = root_client.requests[-1].approve()
    assert isinstance(req_result, SyftSuccess)
    assert root_client.requests[-1].status.value == 2

    all_image_tags = [
        im.image_identifier.full_name_with_tag
        for im in root_client.images.get_all()
        if im.image_identifier
    ]
    tag = (
        worker_config.tag
        if isinstance(worker_config, PrebuiltWorkerConfig)
        else docker_tag
    )
    assert tag in all_image_tags
    launched_pool = root_client.worker_pools["recordlinkage-pool"]
    assert isinstance(launched_pool, WorkerPool)
    assert len(launched_pool.worker_list) == 2


@pytest.mark.parametrize(
    "docker_tag,worker_config,n_images",
    WORKER_CONFIG_TEST_CASES_WITH_N_IMAGES,
)
def test_create_pool_request_accept(
    worker: Worker,
    docker_tag: str,
    worker_config: WorkerConfig,
    n_images: int,
    ds_client: SyftClient,
) -> None:
    """
    Test the functionality of `SyftWorkerPoolService.create_pool_request`
    when the request is accepted
    """
    # construct a root client and data scientist client for a datasite
    root_client = worker.root_client
    assert root_client.credentials != ds_client.credentials

    # the DO submits the docker config to build an image
    submit_result = root_client.api.services.worker_image.submit(
        worker_config=worker_config
    )
    assert isinstance(submit_result, SyftSuccess)
    assert len(root_client.images.get_all()) == n_images

    # The root client builds the image
    worker_image: SyftWorkerImage = root_client.api.services.worker_image.get_by_config(
        worker_config
    )
    if not worker_image.is_prebuilt:
        docker_build_result = root_client.api.services.worker_image.build(
            image_uid=worker_image.id,
            tag=docker_tag,
        )
        # update the worker image variable after the image was built
        worker_image: SyftWorkerImage = (
            root_client.api.services.worker_image.get_by_config(worker_config)
        )
        assert isinstance(docker_build_result, SyftSuccess)
        assert worker_image.image_identifier.full_name_with_tag == docker_tag

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


WORKER_CONFIGS = [test_case[1] for test_case in WORKER_CONFIG_TEST_CASES]


@pytest.mark.parametrize("worker_config", WORKER_CONFIGS)
def test_get_by_worker_config(
    worker: Worker,
    worker_config: WorkerConfig,
) -> None:
    root_client = worker.root_client
    for config in WORKER_CONFIGS:
        root_client.api.services.worker_image.submit(worker_config=config)

    worker_image = root_client.api.services.worker_image.get_by_config(worker_config)
    assert worker_image.config == worker_config
