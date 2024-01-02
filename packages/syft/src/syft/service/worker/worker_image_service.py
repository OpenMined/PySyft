# stdlib
import contextlib
from typing import List
from typing import Optional
from typing import Union

# third party
import docker
import pydantic

# relative
from ...custom_worker.config import DockerWorkerConfig
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.datetime import DateTime
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from .image_registry import SyftImageRegistry
from .image_registry_service import SyftImageRegistryService
from .utils import docker_build
from .utils import docker_push
from .worker_image import SyftWorkerImage
from .worker_image import SyftWorkerImageTag
from .worker_image_stash import SyftWorkerImageStash


@serializable()
class SyftWorkerImageService(AbstractService):
    store: DocumentStore
    stash: SyftWorkerImageStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = SyftWorkerImageStash(store=store)

    @service_method(
        path="worker_image.submit_dockerfile",
        name="submit_dockerfile",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def submit_dockerfile(
        self, context: AuthedServiceContext, docker_config: DockerWorkerConfig
    ) -> Union[SyftWorkerImage, SyftError]:
        worker_image = SyftWorkerImage(
            config=docker_config,
            created_by=context.credentials,
        )
        res = self.stash.set(context.credentials, worker_image)

        if res.is_err():
            return SyftError(message=res.err())

        return worker_image

    @service_method(
        path="worker_image.build",
        name="build",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def build(
        self,
        context: AuthedServiceContext,
        image: UID,
        tag: str,
        version: str = "latest",
        registry: Optional[UID] = None,
    ) -> Union[SyftSuccess, SyftError]:
        registry_obj: Optional[SyftImageRegistry] = None
        image_registry_service: Optional[SyftImageRegistryService] = None

        result = self.stash.get_by_uid(credentials=context.credentials, uid=image)
        if result.is_err():
            return SyftError(
                message=f"Failed to get image for uid: {image}. Error: {result.err()}"
            )
        worker_image: SyftWorkerImage = result.ok()

        if worker_image.built_at:
            return SyftError(message=f"Image<{image}> is already built")

        if registry:
            # get registry from image registry service
            image_registry_service = context.node.get_service(SyftImageRegistryService)
            result = image_registry_service.get_by_id(context, registry)
            if result.is_err():
                return result
            registry_obj = result.ok()

        try:
            if registry_obj:
                image_tag = SyftWorkerImageTag.from_registry(
                    tag=f"{tag}:{version}",
                    registry=registry_obj,
                )
            else:
                image_tag = SyftWorkerImageTag.from_str(f"{tag}:{version}")
        except pydantic.ValidationError as e:
            return SyftError(message=f"Failed to create tag: {e}")

        if not context.node.in_memory_workers:
            worker_image.image_tag = image_tag
            result = docker_build(image)

            if isinstance(result, SyftError):
                return result

            (image, logs) = result
            worker_image.built_at = DateTime.now()
            worker_image.image_hash = image.id

            result = SyftSuccess(message=f"Build {worker_image} succeeded.\n{logs}")
        else:
            result = SyftSuccess(
                message="Image building skipped, since using InMemory workers."
            )

        update_result = self.stash.update(context.credentials, obj=worker_image)

        if update_result.is_err():
            return SyftError(
                message=f"Failed to update image meta information: {update_result.err()}"
            )

        return result

    @service_method(
        path="worker_image.push",
        name="push",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def push(
        self,
        context: AuthedServiceContext,
        image: UID,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(credentials=context.credentials, uid=image)
        if result.is_err():
            return SyftError(
                message=f"Failed to get image for uid: {image}. Error: {result.err()}"
            )
        worker_image: SyftWorkerImage = result.ok()

        if worker_image.built_at is None:
            return SyftError(
                message=f"Image {worker_image} is not built yet. Please build it first."
            )

        result = docker_push(
            image=worker_image,
            username=username,
            password=password,
        )
        return SyftSuccess(
            message=f'The image was successfully pushed to "{worker_image.image_tag.full_tag}"'
        )

    @service_method(
        path="worker_image.get_all",
        name="get_all",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[List[SyftWorkerImage], SyftError]:
        """
        One image one docker file for now
        TODO: change repr when listing
        """
        result = self.stash.get_all(credentials=context.credentials)
        if result.is_err():
            return SyftError(message=f"{result.err()}")

        return result.ok()

    @service_method(
        path="worker_image.delete",
        name="delete",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def delete(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        #  Delete Docker image given image tag
        res = self.stash.get_by_uid(credentials=context.credentials, uid=uid)
        if res.is_err():
            return SyftError(message=f"{res.err()}")

        image: SyftWorkerImage = res.ok()

        if image and image.image_tag:
            try:
                full_tag: str = image.image_tag.full_tag
                with contextlib.closing(docker.from_env()) as client:
                    client.images.remove(image=full_tag)
            except docker.errors.ImageNotFound:
                return SyftError(message=f"Image {full_tag} not found.")
            except Exception as e:
                return SyftError(
                    message=f"Failed to delete image {full_tag}. Error: {e}"
                )

        result = self.stash.delete_by_uid(credentials=context.credentials, uid=uid)

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        returned_message: str = (
            result.ok().message + f". Image {uid} deleted successfully."
        )

        return SyftSuccess(message=returned_message)


TYPE_TO_SERVICE[SyftWorkerImage] = SyftWorkerImageService
SERVICE_TO_TYPES[SyftWorkerImageService].update({SyftWorkerImage})
