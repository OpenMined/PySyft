# stdlib
import io
import json
from typing import Iterator
from typing import Optional

# third party
import docker
from typing_extensions import Self

# relative
from ...custom_worker.config import DockerWorkerConfig
from ...custom_worker.config import WorkerConfig
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...types.base import SyftBaseModel
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..response import SyftError
from ..response import SyftSuccess
from .image_registry import SyftImageRegistry


def parse_output(log_iterator: Iterator) -> str:
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


@serializable()
class SyftWorkerImageTag(SyftBaseModel):
    registry: Optional[SyftImageRegistry | UID]
    repo: str
    tag: str

    @classmethod
    def from_str(cls, full_str: str) -> Self:
        repo_url, tag = full_str.rsplit(":", 1)
        args = repo_url.rsplit("/", 2)

        if len(args) == 3:
            registry = SyftImageRegistry.from_url(args[0])
            repo = "/".join(args[1:])
        else:
            registry = None
            repo = "/".join(args)
        return cls(repo=repo, registry=registry, tag=tag)

    @property
    def full_tag(self) -> str:
        return f"{self.repo}:{self.tag}"

    def __hash__(self) -> int:
        return hash(self.repo + self.tag + str(hash(self.registry)))


@serializable()
class SyftWorkerImage(SyftObject):
    __canonical_name__ = "SyftWorkerImage"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["config"]
    __attr_searchable__ = ["image_tag", "image_hash", "created_by"]

    id: UID
    config: WorkerConfig
    image_tag: Optional[SyftWorkerImageTag]
    image_hash: Optional[str]
    created_at: DateTime = DateTime.now()
    created_by: SyftVerifyKey


def build_using_docker(
    client: docker.DockerClient, worker_image: SyftWorkerImage, push: bool = True
):
    if not isinstance(worker_image.config, DockerWorkerConfig):
        # Handle this to worker with CustomWorkerConfig later
        return SyftError("We only support DockerWorkerConfig")

    try:
        file_obj = io.BytesIO(worker_image.config.dockerfile.encode("utf-8"))

        # docker build -f <dockerfile> <buildargs> <path>
        result = client.images.build(
            fileobj=file_obj,
            rm=True,
            tag=worker_image.image_tag.full_tag,
        )
        worker_image.image_hash = result[0].id
        log = parse_output(result[1])
        return worker_image, SyftSuccess(
            message=f"Build {worker_image} succeeded.\n{log}"
        )
    except docker.errors.BuildError as e:
        return worker_image, SyftError(message=f"Failed to build {worker_image}. {e}")
