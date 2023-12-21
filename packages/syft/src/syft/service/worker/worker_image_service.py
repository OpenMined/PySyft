# stdlib
import contextlib
import json
import os
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import docker
import pydantic

# relative
from ...custom_worker.builder import CustomWorkerBuilder
from ...custom_worker.builder import Image
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
from .worker_image import SyftWorkerImage
from .worker_image import SyftWorkerImageTag
from .worker_image_stash import SyftWorkerImageStash


@serializable(without=["builder"])
class SyftWorkerImageService(AbstractService):
    store: DocumentStore
    stash: SyftWorkerImageStash
    builder: CustomWorkerBuilder

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = SyftWorkerImageStash(store=store)
        self.builder = CustomWorkerBuilder()

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
        worker_image: SyftWorkerImage = None
        registry_obj: Optional[SyftImageRegistry] = None
        image_registry_service: Optional[SyftImageRegistryService] = None

        result = self.stash.get_by_uid(credentials=context.credentials, uid=image)
        if result.is_err():
            return SyftError(
                message=f"Failed to get image for uid: {image}. Error: {result.err()}"
            )
        worker_image = result.ok()

        if worker_image.built_on:
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

        worker_image.image_tag = image_tag

        result = self.docker_build(image=worker_image)

        if isinstance(result, SyftError):
            return result

        (image, logs) = result
        worker_image.built_on = DateTime.now()
        worker_image.image_hash = image.id

        update_result = self.stash.update(context.credentials, obj=worker_image)

        if update_result.is_err():
            return SyftError(
                message=f"Failed to update image meta information: {update_result.err()}"
            )

        return SyftSuccess(message=f"Build {worker_image} succeeded.\n{logs}")

    @service_method(
        path="worker_image.push",
        name="push",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def push(
        self,
        context: AuthedServiceContext,
        image: UID,
        username: str = "",
        password: str = "",
    ) -> Union[SyftSuccess, SyftError]:
        worker_image: SyftWorkerImage = None
        result = self.stash.get_by_uid(credentials=context.credentials, uid=image)
        if result.is_err():
            return SyftError(
                message=f"Failed to get image for uid: {image}. Error: {result.err()}"
            )
        worker_image = result.ok()

        if worker_image.built_on is None:
            return SyftError(
                message=f"Image {worker_image} is not built yet. Please build it first."
            )

        result = self.docker_push(
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

    def docker_build(
        self, image: SyftWorkerImage
    ) -> Union[Tuple[Image, Iterable[str]], SyftError]:
        try:
            (image, logs) = self.builder.build_image(
                config=image.config,
                tag=image.image_tag.full_tag,
            )
            parsed_logs = self.parse_output(logs)
            return (image, parsed_logs)
        except docker.errors.APIError as e:
            return SyftError(
                message=f"Docker API error when building {image.image_tag}. Reason - {e}"
            )
        except docker.errors.DockerException as e:
            return SyftError(
                message=f"Docker exception when building {image.image_tag}. Reason - {e}"
            )
        except Exception as e:
            return SyftError(
                message=f"Unknown exception when building {image.image_tag}. Reason - {e}"
            )

    def docker_push(
        self,
        image: SyftWorkerImage,
        username: str = "",
        password: str = "",
    ) -> List[str]:
        try:
            result = self.builder.push_image(
                tag=image.image_tag.full_tag,
                registry_url=image.image_tag.registry_host,
                username=username,
                password=password,
            )

            parsed_result = result.split(os.linesep)

            if "error" in result:
                result = SyftError(
                    message=f"Failed to push {image.image_tag}. Logs - {parsed_result}"
                )

            return parsed_result
        except docker.errors.APIError as e:
            return SyftError(
                message=f"Docker API error when pushing {image.image_tag}. {e}"
            )
        except docker.errors.DockerException as e:
            return SyftError(
                message=f"Docker exception when pushing {image.image_tag}. Reason - {e}"
            )
        except Exception as e:
            return SyftError(
                message=f"Unknown exception when pushing {image.image_tag}. Reason - {e}"
            )

    def parse_output(self, log_iterator: Iterator) -> str:
        log = ""
        for line in log_iterator:
            for item in line.values():
                if isinstance(item, str):
                    log += item
                elif isinstance(item, dict):
                    log += json.dumps(item)
                else:
                    log += str(item)
        return log


TYPE_TO_SERVICE[SyftWorkerImage] = SyftWorkerImageService
SERVICE_TO_TYPES[SyftWorkerImageService].update({SyftWorkerImage})
