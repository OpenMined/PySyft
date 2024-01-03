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
from ...types.dicttuple import DictTuple
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from .worker_image import SyftWorkerImage
from .worker_image import SyftWorkerImageIdentifier
from .worker_image import build_using_docker
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
        uid: UID,
        tag: str,
        push: bool = False,
        container_registry: Optional[str] = None,
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(credentials=context.credentials, uid=uid)
        if result.is_err():
            return SyftError(
                message=f"Failed to get image for uid: {uid}. Error: {result.err()}"
            )

        worker_image: SyftWorkerImage = result.ok()

        try:
            image_identifier: (
                SyftWorkerImageIdentifier
            ) = SyftWorkerImageIdentifier.from_str(full_str=tag)
        except pydantic.ValidationError as e:
            return SyftError(message=f"Failed to create tag: {e}")

        worker_image.image_identifier = image_identifier

        if not context.node.in_memory_workers:
            with contextlib.closing(docker.from_env()) as client:
                worker_image, result = build_using_docker(
                    client=client,
                    worker_image=worker_image,
                    push=push,
                    dev_mode=context.node.dev_mode,
                )

            if isinstance(result, SyftError):
                return result
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
