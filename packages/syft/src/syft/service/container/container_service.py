# stdlib
import io
import json
from typing import Any
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union

# third party
import docker
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .container import ContainerCommand
from .container import ContainerImage

ImageNamePartitionKey = PartitionKey(key="name", type_=str)
CommandNamePartitionKey = PartitionKey(key="name", type_=str)


@serializable()
class ContainerCommandAdded(SyftSuccess):
    pass


def parse_output(log_iterator: Iterator) -> str:
    log = ""
    for line in log_iterator:
        for item in list(line.values()):
            if isinstance(item, str):
                log += item
            elif isinstance(item, dict):
                log += json.dumps(item)
            else:
                log += str(item)
    return log


def build_image(container_image: ContainerImage) -> Union[SyftSuccess, SyftError]:
    try:
        client = docker.from_env()
        f = io.BytesIO(container_image.dockerfile.encode("utf-8"))
        result = client.images.build(fileobj=f, rm=True, tag=container_image.tag)
        log = parse_output(result[1])
        return SyftSuccess(message=f"Build {container_image} succeeded.\n{log}")
    except docker.errors.BuildError as e:
        return SyftError(message=f"Failed to build {container_image}. {e}")


@instrument
@serializable()
class ContainerImageStash(BaseUIDStoreStash):
    object_type = ContainerImage
    settings: PartitionSettings = PartitionSettings(
        name=ContainerImage.__canonical_name__, object_type=ContainerImage
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[Optional[ContainerImage], str]:
        qks = QueryKeys(qks=[ImageNamePartitionKey.with_obj(name)])
        return self.query_one(credentials, qks=qks)

    def add(
        self, credentials: SyftVerifyKey, image: ContainerImage
    ) -> Result[ContainerImage, str]:
        res = self.check_type(image, ContainerImage)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().set(credentials=credentials, obj=res.ok())


@instrument
@serializable()
class ContainerCommandStash(BaseUIDStoreStash):
    object_type = ContainerCommand
    settings: PartitionSettings = PartitionSettings(
        name=ContainerCommand.__canonical_name__, object_type=ContainerCommand
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[Optional[ContainerCommand], str]:
        qks = QueryKeys(qks=[CommandNamePartitionKey.with_obj(name)])
        return self.query_one(credentials, qks=qks)

    def add(
        self, credentials: SyftVerifyKey, command: ContainerCommand
    ) -> Result[ContainerCommand, str]:
        res = self.check_type(command, ContainerCommand)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().set(credentials=credentials, obj=res.ok())


@instrument
@serializable()
class ContainerService(AbstractService):
    store: DocumentStore
    stash: ContainerImageStash
    command_stash: ContainerCommandStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = ContainerImageStash(store=store)
        self.command_stash = ContainerCommandStash(store=store)

    @service_method(path="container.add_image", name="add_image", autosplat=["image"])
    def add_image(
        self, context: AuthedServiceContext, image: ContainerImage
    ) -> Union[SyftSuccess, SyftError]:
        """Add a Container Image."""

        result = self.stash.add(context.credentials, image=image)
        if result.is_ok():
            return SyftSuccess(message=f"ContainerImage added: {image}")
        return SyftError(
            message=f"Failed to add ContainerImage {image}. {result.err()}"
        )

    @service_method(path="container.build_image", name="build_image")
    def build_image(
        self, context: AuthedServiceContext, name: str
    ) -> Union[SyftSuccess, SyftError]:
        """Build a Container Image."""

        result = self.stash.get_by_name(context.credentials, name=name)
        if result.is_ok():
            if not result.ok():
                return SyftError(message=f"ContainerImage {name} does not exist.")
            image = result.ok()
            result = build_image(container_image=image)
            return result
        return SyftError(
            message=f"Failed to build ContainerImage {name}. {result.err()}"
        )

    @service_method(path="container.get_images", name="get_images")
    def get_images(
        self, context: AuthedServiceContext
    ) -> Union[List[ContainerImage], SyftError]:
        results = self.stash.get_all(context.credentials)
        if results.is_ok():
            return results.ok()
        return SyftError(messages="Unable to get ContainerImages")

    @service_method(path="container.add_command", name="add_command")
    def add_command(
        self, context: AuthedServiceContext, command: ContainerCommand
    ) -> Union[ContainerCommandAdded, SyftError]:
        """Register a ContainerCommand."""

        result = self.command_stash.add(context.credentials, command=command)
        if result.is_ok():
            return ContainerCommandAdded(message=f"ContainerCommand added: {command}")
        return SyftError(
            message=f"Failed to add ContainerCommand {command}. {result.err()}"
        )

    @service_method(path="container.get_commands", name="get_commands")
    def get_commands(
        self, context: AuthedServiceContext
    ) -> Union[List[ContainerCommand], SyftError]:
        results = self.command_stash.get_all(context.credentials)
        if results.is_ok():
            return results.ok()
        return SyftError(messages="Unable to get ContainerCommands")

    @service_method(path="container.call", name="call", roles=GUEST_ROLE_LEVEL)
    def call(
        self,
        context: AuthedServiceContext,
        image_name: str,
        command_name: str,
        **kwargs: Any,
    ) -> Union[SyftSuccess, SyftError]:
        """Call a ContainerCommand"""

        result = self.stash.get_by_name(context.credentials, name=image_name)
        if result.is_ok():
            if not result.ok():
                return SyftError(message=f"ContainerImage {image_name} does not exist.")
        image = result.ok()

        result = self.command_stash.get_by_name(context.credentials, name=command_name)
        if result.is_ok():
            if not result.ok():
                return SyftError(
                    message=f"ContainerCommand {command_name} does not exist."
                )

        command = result.ok()

        print("Got image and command", image, command)
