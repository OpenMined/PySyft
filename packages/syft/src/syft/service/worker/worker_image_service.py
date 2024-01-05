# stdlib
import contextlib
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import docker
import pydantic

# relative
from ...custom_worker.config import DockerWorkerConfig
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.datetime import DateTime
from ...types.dicttuple import DictTuple
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from .image_registry import SyftImageRegistry
from .image_registry_service import SyftImageRegistryService
from .utils import docker_build
from .utils import docker_push
from .worker_image import SyftWorkerImage
from .worker_image import SyftWorkerImageIdentifier
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
    ) -> Union[SyftSuccess, SyftError]:
        image_identifier = SyftWorkerImageIdentifier(repo="", tag="")
        worker_image = SyftWorkerImage(
            config=docker_config,
            created_by=context.credentials,
            image_identifier=image_identifier,
        )
        res = self.stash.set(context.credentials, worker_image)

        if res.is_err():
            return SyftError(message=res.err())

        return SyftSuccess(
            message=f"Dockerfile <id: {worker_image.id}> successfully submitted."
        )

    @service_method(
        path="worker_image.build",
        name="build",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def build(
        self,
        context: AuthedServiceContext,
        image_uid: UID,
        tag: str,
        registry_uid: Optional[UID] = None,
    ) -> Union[SyftSuccess, SyftError]:
        registry: SyftImageRegistry = None

        result = self.stash.get_by_uid(credentials=context.credentials, uid=image_uid)
        if result.is_err():
            return SyftError(
                message=f"Failed to get image for uid: {image_uid}. Error: {result.err()}"
            )

        worker_image: SyftWorkerImage = result.ok()

        if registry_uid:
            # get registry from image registry service
            image_registry_service: SyftImageRegistryService = context.node.get_service(
                SyftImageRegistryService
            )
            result = image_registry_service.get_by_id(context, registry_uid)
            if result.is_err():
                return result
            registry: SyftImageRegistry = result.ok()

        try:
            if registry:
                image_identifier = SyftWorkerImageIdentifier.with_registry(
                    tag=tag, registry=registry
                )
            else:
                image_identifier = SyftWorkerImageIdentifier.from_str(tag=tag)
        except pydantic.ValidationError as e:
            return SyftError(message=f"Failed to create tag: {e}")

        # if image is already built and identifier is unchanged, return an error
        if (
            worker_image.built_at
            and worker_image.image_identifier
            and worker_image.image_identifier.full_name_with_tag
            == image_identifier.full_name_with_tag
        ):
            return SyftError(message=f"Image<{image_uid}> is already built")

        worker_image.image_identifier = image_identifier

        if not context.node.in_memory_workers:
            result = docker_build(worker_image)
            if isinstance(result, SyftError):
                return result

            worker_image.image_hash = result.image_hash
            worker_image.built_at = DateTime.now()

            result = SyftSuccess(
                message=f"Build {worker_image} succeeded.\n{result.logs}"
            )
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
        if context.node.in_memory_workers:
            return SyftSuccess(
                message="Skipped pushing image, since using InMemory workers."
            )

        result = self.stash.get_by_uid(credentials=context.credentials, uid=image)
        if result.is_err():
            return SyftError(
                message=f"Failed to get image for uid: {image}. Error: {result.err()}"
            )
        worker_image: SyftWorkerImage = result.ok()

        if (
            worker_image.image_identifier is None
            or worker_image.image_identifier.registry_host == ""
        ):
            return SyftError(
                message=f"Image {worker_image} does not have a valid registry host."
            )
        elif worker_image.built_at is None:
            return SyftError(
                message=f"Image {worker_image} is not built yet. Please build it first."
            )

        result = docker_push(
            image=worker_image,
            username=username,
            password=password,
        )

        if isinstance(result, SyftError):
            return result

        return SyftSuccess(
            message=f'The image was successfully pushed to "{worker_image.image_identifier.full_name_with_tag}"'
        )

    @service_method(
        path="worker_image.get_all",
        name="get_all",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[DictTuple[str, SyftWorkerImage], SyftError]:
        """
        One image one docker file for now
        """
        result = self.stash.get_all(credentials=context.credentials)
        if result.is_err():
            return SyftError(message=f"{result.err()}")
        images: List[SyftWorkerImage] = result.ok()

        res: List[Tuple] = []
        for im in images:
            if im.image_identifier is not None:
                res.append((im.image_identifier.repo_with_tag, im))
            else:
                res.append(("default-worker-image", im))

        return DictTuple(res)

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

        if not context.node.in_memory_workers and image and image.image_identifier:
            try:
                full_tag: str = image.image_identifier.repo_with_tag
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
