# third party
import pytest

# syft absolute
import syft as sy
from syft.client.domain_client import DomainClient
from syft.custom_worker.config import DockerWorkerConfig
from syft.service.response import SyftSuccess

DOCKER_CONFIG_OPENDP = """
    FROM openmined/grid-backend:0.8.4-beta.12
    RUN pip install opendp
"""


@pytest.mark.container_workload
def test_image_build(domain_1_port) -> None:
    domain_client: DomainClient = sy.login(
        port=domain_1_port, email="info@openmined.org", password="changethis"
    )

    # Submit Docker Worker Config
    docker_config = DockerWorkerConfig(dockerfile=DOCKER_CONFIG_OPENDP)

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
    docker_tag = f"openmined/custom-worker-opendp:{tag_version}"
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
    assert len(domain_client.worker_pools.get_all()) == 1

    # Submit Docker Worker Config
    docker_config = DockerWorkerConfig(dockerfile=DOCKER_CONFIG_OPENDP)

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
    assert len(domain_client.worker_pools.get_all()) == 2

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

    # Delete the worker pool and its workers
    for worker in worker_pool.workers:
        res = domain_client.api.services.worker.delete(uid=worker.id, force=True)
        assert isinstance(res, sy.SyftSuccess)

    # Clean the build images
    delete_result = domain_client.api.services.worker_image.remove(uid=worker_image.id)
    assert isinstance(delete_result, sy.SyftSuccess)


# TODO: register ds client, ds requests to create an image and pool creation,
# do approves, then ds creates a function attached to the worker pool, then creates another
# request. DO approves and runs the function
