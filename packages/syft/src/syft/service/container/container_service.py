# stdlib
import io
import json
import os
import tempfile
from typing import Any
from typing import Dict
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
from ...types.file import SyftFile
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .container import ContainerCommand
from .container import ContainerImage
from .container import ContainerResult

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


def run_command(
    container_image: ContainerImage,
    command: ContainerCommand,
    user_kwargs: Dict[str, SyftFile],
    files: Dict[str, SyftFile],
) -> ContainerResult:
    client = docker.from_env()

    # create temporary sandbox file system
    temp_dirs = []

    # ... do stuff with dirpath
    temp_dir = tempfile.mkdtemp()
    # temp_dir = tempfile.TemporaryDirectory()
    temp_dirs.append(temp_dir)
    volumes = {temp_dir: {"bind": "/sandbox", "mode": "rw"}}

    for container_mount in container_image.volumes:
        # print("got container_mount", container_mount, type(container_mount))
        volume = client.volumes.create(name=container_mount.name)
        # print("got volume", volume, type(volume))
        volumes[volume.name] = {
            "bind": container_mount.internal_mountpath,
            "mode": container_mount.mode,
        }

    for container_mount in command.mounts:
        # mount_dir = tempfile.TemporaryDirectory()
        mount_dir = tempfile.mkdtemp()
        temp_dirs.append(mount_dir)
        container_mount.file.write_file(path=mount_dir)
        local_path = f"{mount_dir}/{container_mount.file.filename}"
        os.chmod(local_path, int(container_mount.unix_permission, base=8))

        mount = {
            "bind": container_mount.internal_filepath,
            "mode": container_mount.mode,
        }
        volumes[local_path] = mount
        # print("local path", local_path, "mount", mount)

    # start container
    container = None
    try:
        # try:
        #     container = client.containers.get(container_name)
        #     container.stop()
        #     container = None
        # except Exception: # nosec
        #     pass

        container = client.containers.run(
            container_image.tag,
            # name=container_name,
            volumes=volumes,
            command="sleep 999999",
            detach=True,
            auto_remove=True,
        )

        result = container.exec_run(
            cmd="bash /start.sh || true", stdout=True, stderr=True, demux=True
        )
        # print("result of start.sh", result)

        # write the files
        for _k, v in files.items():
            if isinstance(v, list):
                for sub_v in v:
                    write_result = sub_v.write_file(path=temp_dir)
                    if not write_result:
                        return write_result
            else:
                write_result = v.write_file(path=temp_dir)
                if not write_result:
                    return write_result

        # print("user_kwargs", user_kwargs)
        # print("files", files)
        cmd = command.cmd(run_user_kwargs=user_kwargs, run_files=files)
        print("> running cmd", cmd)
        result = container.exec_run(cmd=cmd, stdout=True, stderr=True, demux=True)
        container_result = ContainerResult.from_execresult(result=result)
        container_result.image_name = container_image.name
        container_result.image_tag = container_image.tag
        container_result.command_name = command.name
        if command.return_filepath:
            return_file = SyftFile.from_path(f"{temp_dir}/{command.return_filepath}")
            container_result.return_file = return_file
        return container_result
    except Exception as e:
        print(f"Failed to run command in container. {command} {container_image}. {e}")
    finally:
        if container:
            pass
            # container.stop()
        for _ in temp_dirs:
            pass
            # print("remove temp_dir", temp_dir)
            # shutil.rmtree(temp_dir)


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
        # TODO: Add ability to specify which roles see which endpoints
        # for now skip auth
        # results = self.command_stash.get_all(context.credentials)
        results = self.command_stash.get_all(context.node.verify_key)
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

        result = self.stash.get_by_name(context.node.verify_key, name=image_name)
        if result.is_ok():
            if not result.ok():
                return SyftError(message=f"ContainerImage {image_name} does not exist.")
        image = result.ok()

        result = self.command_stash.get_by_name(
            context.node.verify_key, name=command_name
        )
        if result.is_ok():
            if not result.ok():
                return SyftError(
                    message=f"ContainerCommand {command_name} does not exist."
                )

        command = result.ok()

        files = {}
        user_kwargs = {}
        for k, v in kwargs.items():
            key = k
            for possible_keys in command.user_kwargs:
                # some cli args are - but only _ allowed in python signatures
                if "-" in possible_keys and possible_keys.replace("-", "_") == key:
                    key = key.replace("_", "-")

            # todo make another SyftList object or some safe way to check them all
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], SyftFile):
                for item in v:
                    if not isinstance(item, SyftFile):
                        raise Exception("All files in a list must be files")
                files[key] = v

            if isinstance(v, SyftFile):
                files[key] = v
            else:
                user_kwargs[key] = v
        try:
            result = run_command(
                container_image=image,
                command=command,
                user_kwargs=user_kwargs,
                files=files,
            )
            return result
        except Exception as e:
            return SyftError(f"Failed to run command. {e}")
