# syft absolute
import syft as sy
from syft.custom_worker.config import DockerWorkerConfig
from syft.server.worker import Worker
from syft.service.response import SyftSuccess
from syft.service.worker.worker_image import SyftWorkerImage
from syft.types.datetime import DateTime


def get_docker_config():
    # the DS makes a request to create an image and a pool based on the image
    custom_dockerfile = f"""
        FROM openmined/syft-backend:{sy.__version__}
        RUN pip install recordlinkage
    """
    return DockerWorkerConfig(dockerfile=custom_dockerfile)


def test_syft_worker(worker: Worker):
    """
    Test the functionalities of `SyftWorkerImageService.build`,
    `SyftWorkerPoolService.launch`, and the `SyftWorker` class
    """
    root_client = worker.root_client
    docker_config = get_docker_config()
    submit_result = root_client.api.services.worker_image.submit(
        worker_config=docker_config
    )
    assert isinstance(submit_result, SyftSuccess)

    # The root client builds the image
    for im in root_client.images:
        if im.config == docker_config:
            worker_image: SyftWorkerImage = im
    docker_tag = "openmined/custom-worker-opendp:latest"
    docker_build_result = root_client.api.services.worker_image.build(
        image_uid=worker_image.id,
        tag=docker_tag,
    )
    assert isinstance(docker_build_result, SyftSuccess)

    # Launch a pool
    pool_name = "custom-worker-pool"
    num_workers = 3
    worker_pool_res = root_client.api.services.worker_pool.launch(
        pool_name=pool_name,
        image_uid=worker_image.id,
        num_workers=num_workers,
    )
    assert worker_pool_res
    assert len(worker_pool_res) == num_workers
    assert len(root_client.worker_pools.get_all()) == 2
    for pool in root_client.worker_pools:
        if pool.name == pool_name:
            worker_pool = pool
    assert len(worker_pool.worker_list) == num_workers
    for i, worker in enumerate(worker_pool.workers):
        assert worker.name == pool_name + "-" + str(i + 1)
        assert isinstance(worker.logs, str)
        assert worker.worker_pool_name == pool_name
        assert isinstance(worker.created_at, DateTime)
        assert worker.job_id is None
