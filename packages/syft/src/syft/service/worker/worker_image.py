# stdlib
from datetime import datetime
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator
from typing import Optional

# third party
import docker
import pydantic

# relative
from ...custom_worker.config import DockerWorkerConfig
from ...custom_worker.config import WorkerConfig
from ...node.credentials import SyftVerifyKey
from ...types.base import SyftBaseModel
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..response import SyftError
from ..response import SyftSuccess


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


class ContainerImageRegistry(SyftBaseModel):
    url: str
    tls_enabled: bool

    def from_url(cls, full_str: str):
        tls_enabled = True if "https" in full_str else False
        return cls(url=full_str, tls_enabled=tls_enabled)


class SyftWorkerImageTag(SyftBaseModel):
    registry: Optional[ContainerImageRegistry]
    repo: str
    tag: str

    @classmethod
    def from_str(cls, full_str: str):
        repo_url, tag = full_str.rsplit(":", 1)
        args = repo_url.rsplit("/", 2)

        if len(args) == 3:
            registry = ContainerImageRegistry.from_url(args[0])
            repo = "/".join(args[1::])
        else:
            registry = None
            repo = "/".join(args)
        return cls(repo=repo, registry=registry, tag=tag)


class SyftWorkerImage(SyftObject):
    __canonical_name__ = "SyftWorkerImage"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["image_tag"]
    __attr_searchable__ = ["image_tag", "hash", "created_by"]

    id: UID
    config: WorkerConfig
    image_tag: SyftWorkerImageTag
    hash: str
    created_at: datetime
    created_by: SyftVerifyKey


class BuildContext:
    path: Optional[Path]

    @pydantic.validator("path")
    def val_path(cls, v: Optional[str]):
        if v is None:
            temp_dir = TemporaryDirectory()
            v = temp_dir.name
        if isinstance(v, str):
            v = Path(v)

        return v

    def build_from_url(url: str):
        pass

    def from_local_context(url: str):
        pass


def save_dockerfile_to_path(path: Path, file_name: str, data: str):
    full_path = path / file_name
    with open(full_path, "w") as fp:
        fp.write(data)


def build_using_docker(
    build_context: BuildContext, worker_image: SyftWorkerImage, pull: bool = True
):
    if not isinstance(worker_image.config, DockerWorkerConfig):
        # Handle this to worker with CustomWorkerConfig later
        return SyftError("We only support DockerWorkerConfig")

    try:
        client = docker.from_env()
        file_obj = io.BytesIO(worker_image.config.dockerfile.encode("utf-8"))

        # docker build -f <dockerfile> <buildargs> <path>
        result = client.images.build(
            path=build_context.path,
            fileobj=file_obj,
            rm=True,
            tag=worker_image.image_tag.tag,
        )
        log = parse_output(result[1])
        return SyftSuccess(message=f"Build {worker_image} succeeded.\n{log}")
    except docker.errors.BuildError as e:
        return SyftError(message=f"Failed to build {worker_image}. {e}")
