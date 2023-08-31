# stdlib
from functools import cached_property

# third party
from packaging.specifiers import SpecifierSet
from packaging.specifiers import Version
from packaging.version import InvalidVersion

# relative
from .syft_repo import SyftRepo

__all__ = ["SyftVersion", "InvalidVersion"]


class SyftVersion:
    def __init__(self, version: str):
        self._ver: Version = self._resolve(version)

    @property
    def version(self) -> Version:
        """Returns the underlying Version object"""
        return self._ver

    @property
    def release_tag(self) -> str:
        """Returns the Github release version string (e.g. v0.8.2)"""

        return f"v{self.version}"

    @cached_property
    def docker_tag(self) -> str:
        """Returns the docker version/tag (e.g. 0.8.2-beta.26)"""
        manifest = SyftRepo.get_manifest(self.release_tag)
        return manifest["dockerTag"]

    def match(self, ver_spec: str, prereleases: bool = True) -> bool:
        _spec = SpecifierSet(ver_spec, prereleases=prereleases)
        return _spec.contains(self.version)

    def valid_version(self) -> bool:
        return self.release_tag in SyftRepo.all_versions()

    def _resolve(self, version: str) -> Version:
        if version == "latest":
            version = SyftRepo.latest_version()
        if version == "latest-beta":
            version = SyftRepo.latest_version(beta=True)

        return Version(version)

    def __str__(self) -> str:
        return str(self._ver)
