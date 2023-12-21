# stdlib
from typing import Optional
from typing import Tuple
from typing import Union

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
from .image_registry import SyftImageRegistry


@serializable()
class SyftWorkerImageTag(SyftBaseModel):
    repo: str
    tag: str
    registry: Optional[Union[SyftImageRegistry, str]]

    @classmethod
    def from_registry(cls, tag: str, registry: SyftImageRegistry) -> Self:
        """Build a SyftWorkerImageTag from Docker tag & a previously created SyftImageRegistry object."""
        registry_str, repo, tag = SyftWorkerImageTag.parse_str(tag)

        # if we parsed a registry string, make sure it matches the registry object
        if registry_str and registry_str != registry.url:
            raise ValueError(f"Registry URL mismatch: {registry_str} != {registry.url}")

        return cls(repo=repo, tag=tag, registry=registry)

    @classmethod
    def from_str(cls, tag: str) -> Self:
        """Build a SyftWorkerImageTag from a pure-string standard Docker tag."""
        registry, repo, tag = SyftWorkerImageTag.parse_str(tag)
        return cls(repo=repo, registry=registry, tag=tag)

    @staticmethod
    def parse_str(tag: str) -> Tuple[Optional[str], str, str]:
        url, tag = tag.rsplit(":", 1)
        args = url.rsplit("/", 2)

        if len(args) == 3:
            registry = args[0]
            repo = "/".join(args[1:])
        else:
            registry = None
            repo = "/".join(args)

        return registry, repo, tag

    @property
    def full_tag(self) -> str:
        if self.registry is None:
            return f"{self.repo}:{self.tag}"
        return f"{self.registry.url}/{self.repo}:{self.tag}"

    @property
    def registry_host(self) -> str:
        if self.registry is None:
            return ""
        elif isinstance(self.registry, str):
            return self.registry
        else:
            return self.registry.url

    def __hash__(self) -> int:
        return hash(self.repo + self.tag + str(hash(self.registry)))

    def __str__(self) -> str:
        return self.full_tag


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
    built_at: Optional[DateTime]

    def __str__(self) -> str:
        if self.image_hash:
            return f"SyftWorkerImage<{self.id}, {self.image_hash}, {self.built_at}>"
        return f"SyftWorkerImage<{self.id},BUILD PENDING>"
