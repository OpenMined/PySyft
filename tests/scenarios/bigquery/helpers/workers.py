# third party
from unsync import unsync

# syft absolute
import syft as sy
from syft import test_settings


@unsync
async def get_prebuilt_worker_image(events, client, expected_tag, event_name):
    await events.await_for(event_name=event_name, show=True)
    worker_images = client.images.get_all()
    for worker_image in worker_images:
        if expected_tag in str(worker_image.image_identifier):
            assert expected_tag in str(worker_image.image_identifier)
            return worker_image


@unsync
async def create_prebuilt_worker_image(events, client, expected_tag, event_name):
    external_registry = test_settings.get("external_registry", default="docker.io")
    docker_config = sy.PrebuiltWorkerConfig(tag=f"{external_registry}/{expected_tag}")
    result = client.api.services.worker_image.submit(worker_config=docker_config)
    assert isinstance(result, sy.SyftSuccess)
    events.register(event_name)


@unsync
async def add_external_registry(events, client, event_name):
    external_registry = test_settings.get("external_registry", default="docker.io")
    result = client.api.services.image_registry.add(external_registry)
    assert isinstance(result, sy.SyftSuccess)
    events.register(event_name)


@unsync
async def create_worker_pool(
    events, client, worker_pool_name, worker_pool_result, event_name
):
    # block until this is available
    worker_image = worker_pool_result.result(timeout=5)

    result = client.api.services.worker_pool.launch(
        pool_name=worker_pool_name,
        image_uid=worker_image.id,
        num_workers=1,
    )

    if isinstance(result, list) and isinstance(
        result[0], sy.service.worker.worker_pool.ContainerSpawnStatus
    ):
        events.register(event_name)


@unsync
async def check_worker_pool_exists(events, client, worker_pool_name, event_name):
    timeout = 30
    await events.await_for(event_name=event_name, timeout=timeout)
    pools = client.worker_pools.get_all()
    for pool in pools:
        if worker_pool_name == pool.name:
            assert worker_pool_name == pool.name
            return worker_pool_name == pool.name
