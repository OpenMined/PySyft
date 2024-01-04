# stdlib
import io
import json
from typing import Iterator
from typing import Optional

# third party
import docker

# relative
from ...custom_worker.config import DockerWorkerConfig
from ...custom_worker.config import WorkerConfig
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..response import SyftError
from ..response import SyftSuccess
from .image_identifier import SyftWorkerImageIdentifier


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
class SyftWorkerImage(SyftObject):
    __canonical_name__ = "SyftWorkerImage"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["config"]
    __attr_searchable__ = ["config", "image_hash", "created_by"]
    __repr_attrs__ = ["image_identifier", "image_hash", "created_at"]

    id: UID
    config: WorkerConfig
    image_identifier: Optional[SyftWorkerImageIdentifier]
    image_hash: Optional[str]
    created_at: DateTime = DateTime.now()
    created_by: SyftVerifyKey
    built_at: Optional[DateTime]


def build_using_docker(
    client: docker.DockerClient,
    worker_image: SyftWorkerImage,
    push: bool = True,
    dev_mode: bool = False,
):
    if not isinstance(worker_image.config, DockerWorkerConfig):
        # Handle this to worker with CustomWorkerConfig later
        return SyftError("We only support DockerWorkerConfig")

    try:
        file_obj = io.BytesIO(worker_image.config.dockerfile.encode("utf-8"))

        # docker build -f <dockerfile> <buildargs> <path>

        # Enable this once we're able to copy worker_cpu.dockerfile in backend
        # buildargs = {"SYFT_VERSION_TAG": "local-dev"} if dev_mode else {}
        result = client.images.build(
            fileobj=file_obj,
            rm=True,
            tag=worker_image.image_identifier.repo_with_tag,
            forcerm=True,
        )
        worker_image.image_hash = result[0].id
        worker_image.built_at = DateTime.now()
        log = parse_output(result[1])
        return worker_image, SyftSuccess(
            message=f"Build {worker_image} succeeded.\n{log}"
        )
    except docker.errors.BuildError as e:
        return worker_image, SyftError(message=f"Failed to build {worker_image}. {e}")
