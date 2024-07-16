# stdlib

# third party
from typing_extensions import Self

# relative
from ...custom_worker.utils import ImageUtils
from ...serde.serializable import serializable
from ...types.base import SyftBaseModel
from .image_registry import SyftImageRegistry


@serializable(canonical_name="SyftWorkerImageIdentifier", version=1)
class SyftWorkerImageIdentifier(SyftBaseModel):
    """
    Class to identify syft worker images.
    If a user provides an image's identifier with
    "docker.io/openmined/test-nginx:0.7.8", the convention we use for
    image name, tag and repo for now is
        tag = 0.7.8
        repo = openmined/test-nginx
        repo_with_tag = openmined/test-nginx:0.7.8
        full_name = docker.io/openmined/test-nginx
        full_name_with_tag = docker.io/openmined/test-nginx:0.7.8

    References:
        https://docs.docker.com/engine/reference/commandline/tag/#tag-an-image-referenced-by-name-and-tag
    """

    registry: SyftImageRegistry | str | None = None
    repo: str
    tag: str

    __repr_attrs__ = ["registry", "repo", "tag"]

    @classmethod
    def with_registry(cls, tag: str, registry: SyftImageRegistry) -> Self:
        """Build a SyftWorkerImageTag from Docker tag & a previously created SyftImageRegistry object."""
        registry_str, repo, tag = ImageUtils.parse_tag(tag)

        # if we parsed a registry string, make sure it matches the registry object
        if registry_str and registry_str != registry.url:
            raise ValueError(f"Registry URL mismatch: {registry_str} != {registry.url}")

        return cls(repo=repo, tag=tag, registry=registry)

    @classmethod
    def from_str(cls, tag: str) -> Self:
        """Build a SyftWorkerImageTag from a pure-string standard Docker tag."""
        registry, repo, tag = ImageUtils.parse_tag(tag)
        return cls(repo=repo, registry=registry, tag=tag)

    @property
    def repo_with_tag(self) -> str | None:
        if self.repo or self.tag:
            return f"{self.repo}:{self.tag}"
        return None

    @property
    def full_name_with_tag(self) -> str:
        if self.registry is None:
            return f"docker.io/{self.repo}:{self.tag}"
        elif isinstance(self.registry, str):
            return f"{self.registry}/{self.repo}:{self.tag}"
        else:
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
        return self.full_name_with_tag

    def __repr__(self) -> str:
        return f"SyftWorkerImageIdentifier(repo={self.repo}, tag={self.tag}, registry={self.registry})"
