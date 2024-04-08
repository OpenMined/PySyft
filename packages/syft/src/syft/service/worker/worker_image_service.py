# stdlib
import contextlib

# third party
import docker
import pydantic

# relative
from ...custom_worker.config import DockerWorkerConfig
from ...custom_worker.k8s import IN_KUBERNETES
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
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .image_registry import SyftImageRegistry
from .image_registry_service import SyftImageRegistryService
from .utils import image_build
from .utils import image_push
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
    ) -> SyftSuccess | SyftError:
        worker_image = SyftWorkerImage(
            config=docker_config,
            created_by=context.credentials,
        )
        res = self.stash.set(context.credentials, worker_image)

        if res.is_err():
            return SyftError(message=res.err())

        return SyftSuccess(
            message=f"Dockerfile ID: {worker_image.id} successfully submitted."
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
        registry_uid: UID | None = None,
        pull: bool = True,
    ) -> SyftSuccess | SyftError:
        registry: SyftImageRegistry | None = None

        if IN_KUBERNETES and registry_uid is None:
            return SyftError(message="Registry UID is required in Kubernetes mode.")

        result = self.stash.get_by_uid(credentials=context.credentials, uid=image_uid)
        if result.is_err():
            return SyftError(
                message=f"Failed to get image for uid: {image_uid}. Error: {result.err()}"
            )

        worker_image: SyftWorkerImage = result.ok()

        if registry_uid:
            # get registry from image registry service
            image_registry_service: AbstractService = context.node.get_service(
                SyftImageRegistryService
            )
            registry_result = image_registry_service.get_by_id(context, registry_uid)
            if registry_result.is_err():
                return registry_result
            registry = registry_result.ok()

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
            return SyftError(message=f"Image ID: {image_uid} is already built")

        worker_image.image_identifier = image_identifier
        result = None

        if not context.node.in_memory_workers:
            build_result = image_build(worker_image, pull=pull)
            if isinstance(build_result, SyftError):
                return build_result

            worker_image.image_hash = build_result.image_hash
            worker_image.built_at = DateTime.now()

            result = SyftSuccess(
                message=f"Build for Worker ID: {worker_image.id} succeeded.\n{build_result.logs}"
            )
        else:
            result = SyftSuccess(
                message="Image building skipped, since using in-memory workers."
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
        username: str | None = None,
        password: str | None = None,
    ) -> SyftSuccess | SyftError:
        result = self.stash.get_by_uid(credentials=context.credentials, uid=image)
        if result.is_err():
            return SyftError(
                message=f"Failed to get Image ID: {image}. Error: {result.err()}"
            )
        worker_image: SyftWorkerImage = result.ok()

        if not worker_image.is_built:
            return SyftError(message=f"Image ID: {worker_image.id} is not built yet.")
        elif (
            worker_image.image_identifier is None
            or worker_image.image_identifier.registry_host == ""
        ):
            return SyftError(
                message=f"Image ID: {worker_image.id} does not have a valid registry host."
            )

        result = image_push(
            image=worker_image,
            username=username,
            password=password,
        )

        if isinstance(result, SyftError):
            return result

        return SyftSuccess(
            message=f'Pushed Image ID: {worker_image.id} to "{worker_image.image_identifier.full_name_with_tag}".'
        )

    @service_method(
        path="worker_image.get_all",
        name="get_all",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all(
        self, context: AuthedServiceContext
    ) -> DictTuple[str, SyftWorkerImage] | SyftError:
        """
        One image one docker file for now
        """
        result = self.stash.get_all(credentials=context.credentials)
        if result.is_err():
            return SyftError(message=f"{result.err()}")
        images: list[SyftWorkerImage] = result.ok()

        res = {}
        # if image is built, index it by full_name_with_tag
        for im in images:
            if im.is_built and im.image_identifier is not None:
                res[im.image_identifier.full_name_with_tag] = im
        # and then index all images by id
        # TODO: jupyter repr needs to be updated to show unique values
        # (even if multiple keys point to same value)
        res.update({im.id.to_string(): im for im in images if not im.is_built})

        return DictTuple(res)

    @service_method(
        path="worker_image.remove",
        name="remove",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def remove(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        #  Delete Docker image given image tag
        res = self.stash.get_by_uid(credentials=context.credentials, uid=uid)
        if res.is_err():
            return SyftError(message=f"{res.err()}")
        image: SyftWorkerImage = res.ok()

        if context.node.in_memory_workers:
            pass
        elif IN_KUBERNETES:
            # TODO: Implement image deletion in kubernetes
            return SyftError(
                message="Image Deletion is not yet implemented in Kubernetes !!"
            )
        elif image and image.image_identifier:
            try:
                full_tag: str = image.image_identifier.full_name_with_tag
                with contextlib.closing(docker.from_env()) as client:
                    client.images.remove(image=full_tag)
            except docker.errors.ImageNotFound:
                return SyftError(message=f"Image Tag: {full_tag} not found.")
            except Exception as e:
                return SyftError(
                    message=f"Failed to delete Image Tag: {full_tag}. Error: {e}"
                )

        result = self.stash.delete_by_uid(credentials=context.credentials, uid=uid)

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        returned_message: str = (
            result.ok().message + f". Image ID: {uid} deleted successfully."
        )

        return SyftSuccess(message=returned_message)

    @service_method(
        path="worker_image.get_by_uid",
        name="get_by_uid",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftWorkerImage | SyftError:
        res = self.stash.get_by_uid(credentials=context.credentials, uid=uid)
        if res.is_err():
            return SyftError(
                message=f"Failed to get image with uid {uid}. Error: {res.err()}"
            )
        image: SyftWorkerImage = res.ok()
        return image

    @service_method(
        path="worker_image.get_by_config",
        name="get_by_config",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_by_config(
        self, context: AuthedServiceContext, docker_config: DockerWorkerConfig
    ) -> SyftWorkerImage | SyftError:
        res = self.stash.get_by_docker_config(
            credentials=context.credentials, config=docker_config
        )
        if res.is_err():
            return SyftError(
                message=f"Failed to get image with docker config {docker_config}. Error: {res.err()}"
            )
        image: SyftWorkerImage = res.ok()
        return image
