# stdlib
from typing import Optional

# third party
from typing_extensions import Self

# relative
from ...custom_worker.config import WorkerConfig
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...types.base import SyftBaseModel
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID

# from .image_registry import SyftImageRegistry


@serializable()
class SyftWorkerImageTag(SyftBaseModel):
    registry: Optional[str]
    repo: str
    tag: str

    @classmethod
    def from_str(cls, full_str: str) -> Self:
        repo_url, tag = full_str.rsplit(":", 1)
        args = repo_url.rsplit("/", 2)

        if len(args) == 3:
            registry = args[0]
            repo = "/".join(args[1:])
        else:
            registry = None
            repo = "/".join(args)
        return cls(repo=repo, registry=registry, tag=tag)

    @property
    def full_tag(self) -> str:
        if self.registry is None:
            return f"{self.repo}:{self.tag}"
        return f"{self.registry}/{self.repo}:{self.tag}"

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
    built_on: Optional[DateTime]

    def __str__(self) -> str:
        if self.image_hash:
            return f"<SyftWorkerImage {self.id} | {self.image_hash} {self.built_on}>"
        return f"<SyftWorkerImage {self.id} | BUILD PENDING>"
