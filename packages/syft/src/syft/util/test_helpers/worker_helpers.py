# syft absolute
import syft as sy

# relative
from ...client.client import SyftClient
from ...service.response import SyftSuccess
from ...service.worker.worker_image import SyftWorkerImage
from ...types.uid import UID


def build_and_launch_worker_pool_from_docker_str(
    environment: str,
    client: sy.DatasiteClient,
    worker_pool_name: str,
    custom_pool_pod_annotations: dict,
    custom_pool_pod_labels: dict,
    worker_dockerfile: str,
    external_registry: str,
    docker_tag: str,
    scale_to: int,
):
    result = client.api.services.image_registry.add(external_registry)
    assert "success" in result.message  # nosec: B101

    # For some reason, when using k9s, result.value is empty so can't use the below line
    # local_registry = result.value
    local_registry = client.api.services.image_registry[0]

    docker_config = sy.DockerWorkerConfig(dockerfile=worker_dockerfile)
    assert docker_config.dockerfile == worker_dockerfile  # nosec: B101
    submit_result = client.api.services.worker_image.submit(worker_config=docker_config)
    print(submit_result.message)
    assert "success" in submit_result.message  # nosec: B101

    worker_image = submit_result.value

    if environment == "remote":
        docker_build_result = client.api.services.worker_image.build(
            image_uid=worker_image.id,
            tag=docker_tag,
            registry_uid=local_registry.id,
        )
        print(docker_build_result)

    if environment == "remote":
        push_result = client.api.services.worker_image.push(worker_image.id)
        print(push_result)

    result = client.api.services.worker_pool.launch(
        pool_name=worker_pool_name,
        image_uid=worker_image.id,
        num_workers=1,
        pod_annotations=custom_pool_pod_annotations,
        pod_labels=custom_pool_pod_labels,
    )
    print(result)
    # assert 'success' in str(result.message)

    # scale_to > 1 is valid for scale up
    # scale_to = 0 is valid for removing all pods
    # scale_to < 0 should return error from server
    if environment == "remote" and scale_to != 1:
        result = client.worker_pools.scale(number=scale_to, pool_name=worker_pool_name)
        print(result)


def launch_worker_pool_from_docker_tag_and_registry(
    environment: str,
    client: sy.DatasiteClient,
    worker_pool_name: str,
    custom_pool_pod_annotations: dict,
    custom_pool_pod_labels: dict,
    docker_tag: str,
    external_registry: str,
    scale_to: int = 1,
):
    res = client.api.services.image_registry.add(external_registry)
    assert "success" in res.message  # nosec: B101
    docker_config = sy.PrebuiltWorkerConfig(tag=docker_tag)
    image_result = client.api.services.worker_image.submit(worker_config=docker_config)
    assert "success" in res.message  # nosec: B101
    worker_image = image_result.value

    launch_result = client.api.services.worker_pool.launch(
        pool_name=worker_pool_name,
        image_uid=worker_image.id,
        num_workers=1,
        pod_annotations=custom_pool_pod_annotations,
        pod_labels=custom_pool_pod_labels,
    )
    if environment == "remote" and scale_to > 1:
        result = client.worker_pools.scale(number=scale_to, pool_name=worker_pool_name)
        print(result)

    return launch_result


def prune_worker_pool_and_images(client: SyftClient) -> None:
    for pool in client.worker_pools.get_all():
        client.worker_pools.delete(pool.id)

    for image in client.images.get_all():
        client.images.remove(image.id)


def build_and_push_image(
    client: SyftClient,
    image: SyftWorkerImage,
    tag: str,
    registry_uid: UID | None = None,
    reg_username: str | None = None,
    reg_password: str | None = None,
    force_build: bool = False,
) -> None:
    """Build and push the image to the given registry."""
    if image.is_prebuilt:
        return

    build_result = client.api.services.worker_image.build(
        image_uid=image.id, registry_uid=registry_uid, tag=tag, force_build=force_build
    )
    print(build_result.message)

    if isinstance(build_result, SyftSuccess):
        push_result = client.api.services.worker_image.push(
            image.id,
            username=reg_username,
            password=reg_password,
        )
        assert isinstance(push_result, SyftSuccess)  # nosec: B101
        print(push_result.message)
