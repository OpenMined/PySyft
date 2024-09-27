# stdlib
import contextlib

# third party
import docker
import pydantic

# relative
from ...custom_worker.config import PrebuiltWorkerConfig
from ...custom_worker.config import WorkerConfig
from ...custom_worker.k8s import IN_KUBERNETES
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...types.datetime import DateTime
from ...types.dicttuple import DictTuple
from ...types.errors import SyftException
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .image_registry import SyftImageRegistry
from .utils import image_build
from .utils import image_push
from .worker_image import SyftWorkerImage
from .worker_image import SyftWorkerImageIdentifier
from .worker_image_stash import SyftWorkerImageStash


@serializable(canonical_name="SyftWorkerImageService", version=1)
class SyftWorkerImageService(AbstractService):
    stash: SyftWorkerImageStash

    def __init__(self, store: DBManager) -> None:
        self.stash = SyftWorkerImageStash(store=store)

    @service_method(
        path="worker_image.submit",
        name="submit",
        roles=DATA_OWNER_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def submit(
        self, context: AuthedServiceContext, worker_config: WorkerConfig
    ) -> SyftSuccess:
        image_identifier: SyftWorkerImageIdentifier | None = None
        if isinstance(worker_config, PrebuiltWorkerConfig):
            try:
                image_identifier = SyftWorkerImageIdentifier.from_str(worker_config.tag)
            except Exception:
                raise SyftException(
                    public_message=(
                        f"Invalid Docker image name: {worker_config.tag}.\n"
                        + "Please specify the image name in this format <registry>/<repo>:<tag>."
                    )
                )
        worker_image = SyftWorkerImage(
            config=worker_config,
            created_by=context.credentials,
            image_identifier=image_identifier,
        )

        # TODO: I think this was working in python mode due to a bug because
        # it wasn't saying it was duplicate
        # why can we only have a prebuilt or a non prebuilt with the same tag?
        # bigquery uses prebuilt but we need to build and then test that prebuilt works
        # so we kind of need to use one then the other and have it pull from the first
        stored_image = self.stash.set(
            context.credentials, worker_image, ignore_duplicates=True
        ).unwrap()

        return SyftSuccess(
            message=f"Dockerfile ID: {worker_image.id} successfully submitted.",
            value=stored_image,
        )

    @service_method(
        path="worker_image.build",
        name="build",
        roles=DATA_OWNER_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def build(
        self,
        context: AuthedServiceContext,
        image_uid: UID,
        tag: str,
        registry_uid: UID | None = None,
        pull_image: bool = True,
        force_build: bool = False,
    ) -> SyftSuccess:
        registry: SyftImageRegistry | None = None

        if IN_KUBERNETES and registry_uid is None:
            raise SyftException(
                public_message="Registry UID is required in Kubernetes mode."
            )

        worker_image = self.stash.get_by_uid(
            credentials=context.credentials, uid=image_uid
        ).unwrap()
        if registry_uid:
            # get registry from image registry service
            registry = context.server.services.syft_image_registry.get_by_id(
                context, registry_uid
            )

        try:
            if registry:
                image_identifier = SyftWorkerImageIdentifier.with_registry(
                    tag=tag, registry=registry
                )
            else:
                image_identifier = SyftWorkerImageIdentifier.from_str(tag=tag)
        except pydantic.ValidationError as e:
            raise SyftException(public_message=f"Failed to create tag: {e}")

        # if image is already built and identifier is unchanged, return an error
        if (
            worker_image.built_at
            and worker_image.image_identifier
            and worker_image.image_identifier.full_name_with_tag
            == image_identifier.full_name_with_tag
            and not force_build
        ):
            raise SyftException(
                public_message=f"Image ID: {image_uid} is already built"
            )

        worker_image.image_identifier = image_identifier
        result = None

        if not context.server.in_memory_workers:
            build_result = image_build(worker_image, pull=pull_image).unwrap()

            worker_image.image_hash = build_result.image_hash
            worker_image.built_at = DateTime.now()

            result = SyftSuccess(
                message=f"Build for Worker ID: {worker_image.id} succeeded.\n{build_result.logs}"
            )
        else:
            result = SyftSuccess(
                message="Image building skipped, since using in-memory workers."
            )

        self.stash.update(context.credentials, obj=worker_image).unwrap()
        return result

    @service_method(
        path="worker_image.push",
        name="push",
        roles=DATA_OWNER_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def push(
        self,
        context: AuthedServiceContext,
        image_uid: UID,
        username: str | None = None,
        password: str | None = None,
    ) -> SyftSuccess:
        worker_image = self.stash.get_by_uid(
            credentials=context.credentials, uid=image_uid
        ).unwrap()

        if not worker_image.is_built:
            raise SyftException(
                public_message=f"Image ID: {worker_image.id} is not built yet."
            )
        elif (
            worker_image.image_identifier is None
            or worker_image.image_identifier.registry_host == ""
        ):
            raise SyftException(
                public_message=f"Image ID: {worker_image.id} does not have a valid registry host."
            )

        image_push(image=worker_image, username=username, password=password).unwrap()

        return SyftSuccess(
            message=f'Pushed Image ID: {worker_image.id} to "{worker_image.image_identifier.full_name_with_tag}".'
        )

    @service_method(
        path="worker_image.get_all",
        name="get_all",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all(self, context: AuthedServiceContext) -> DictTuple[str, SyftWorkerImage]:
        """
        One image one docker file for now
        """
        images = self.stash.get_all(credentials=context.credentials).unwrap()
        return DictTuple({image.id.to_string(): image for image in images})

    @service_method(
        path="worker_image.remove",
        name="remove",
        roles=DATA_OWNER_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def remove(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess:
        #  Delete Docker image given image tag
        image = self.stash.get_by_uid(credentials=context.credentials, uid=uid).unwrap()

        if context.server.in_memory_workers:
            pass
        elif IN_KUBERNETES:
            # TODO: Implement image deletion in kubernetes
            raise SyftException(
                public_message="Image Deletion is not yet implemented in Kubernetes !!"
            )
        elif image and image.image_identifier:
            try:
                full_tag: str = image.image_identifier.full_name_with_tag
                with contextlib.closing(docker.from_env()) as client:
                    client.images.remove(image=full_tag)
            except docker.errors.ImageNotFound:
                raise SyftException(public_message=f"Image Tag: {full_tag} not found.")
            except Exception as e:
                raise SyftException(
                    public_message=f"Failed to delete Image Tag: {full_tag}. Error: {e}"
                )

        self.stash.delete_by_uid(credentials=context.credentials, uid=uid).unwrap()
        return SyftSuccess(message=f"Image ID: {uid} deleted successfully.")

    @service_method(
        path="worker_image.get_by_uid",
        name="get_by_uid",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_by_uid(self, context: AuthedServiceContext, uid: UID) -> SyftWorkerImage:
        return self.stash.get_by_uid(credentials=context.credentials, uid=uid).unwrap()

    @service_method(
        path="worker_image.get_by_config",
        name="get_by_config",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_by_config(
        self, context: AuthedServiceContext, worker_config: WorkerConfig
    ) -> SyftWorkerImage:
        return self.stash.get_by_worker_config(
            credentials=context.credentials, config=worker_config
        ).unwrap()
