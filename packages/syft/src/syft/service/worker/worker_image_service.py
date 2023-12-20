# stdlib
import json
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union

# third party
import docker
import pydantic

# relative
from ...custom_worker.builder import CustomWorkerBuilder
from ...custom_worker.config import DockerWorkerConfig
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.datetime import DateTime
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
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
        uid: UID,
        tag: str,
        push: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(credentials=context.credentials, uid=uid)
        if result.is_err():
            return SyftError(
                message=f"Failed to get image for uid: {uid}. Error: {result.err()}"
            )

        worker_image: SyftWorkerImage = result.ok()

        try:
            image_tag = SyftWorkerImageTag.from_str(full_str=tag)
        except pydantic.ValidationError as e:
            return SyftError(message=f"Failed to create tag: {e}")

        worker_image.image_tag = image_tag

        result = self.build_using_docker(
            worker_image=worker_image,
            push=push,
            username=username,
            password=password,
        )

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

    def build_using_docker(
        self,
        worker_image: SyftWorkerImage,
        push: bool = True,
        username: str = None,
        password: str = None,
    ) -> Union[tuple, SyftError]:
        if not isinstance(worker_image.config, DockerWorkerConfig):
            # Handle this to worker with CustomWorkerConfig later
            return SyftError("We only support DockerWorkerConfig")

        try:
            builder = CustomWorkerBuilder()
            (image, logs) = builder.build_image(
                config=worker_image.config,
                tag=worker_image.image_tag.full_tag,
            )
            if push:
                # TODO: check for push errors
                push_result = builder.push_image(
                    tag=worker_image.image_tag.full_tag,
                    registry_url=worker_image.image_tag.registry,
                    username=username,
                    password=password,
                )
                if '"error"' in push_result:
                    return SyftError(
                        message=f"Failed to push {worker_image}. {push_result}"
                    )
            logs = self.parse_output(logs)
            return image, logs
        except docker.errors.BuildError as e:
            return SyftError(message=f"Failed to build {worker_image}. {e}")
        except docker.errors.APIError as e:
            return SyftError(
                message=f"Docker API error when building {worker_image}. {e}"
            )
        except Exception as e:
            return SyftError(message=f"Unknown exception occured {worker_image}. {e}")

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
