# stdlib
from hashlib import sha256
from pathlib import Path
from typing import List

# third party
from packaging import version
from pydantic import validator
from typing_extensions import Self
import yaml

# relative
from ..types.base import SyftBaseModel

PYTHON_DEFAULT_VER = "3.11"
PYTHON_MIN_VER = version.parse("3.10")
PYTHON_MAX_VER = version.parse("3.12")


class CustomBuildConfig(SyftBaseModel):
    gpu: bool = False
    # python_version: str = PYTHON_DEFAULT_VER
    python_packages: List[str] = []
    system_packages: List[str] = []
    custom_cmds: List[str] = []

    # @validator("python_version")
    # def validate_python_version(cls, ver: str) -> str:
    #     parsed_ver = version.parse(ver)

    #     # TODO: Check if Wolfi OS/apk supports minor version of python
    #     if parsed_ver.micro != 0:
    #         raise ValueError("Provide only major.minor version of python")

    #     if PYTHON_MIN_VER <= parsed_ver < PYTHON_MAX_VER:
    #         return ver
    #     else:
    #         raise ValueError(
    #             f"Python version must be between {PYTHON_MIN_VER} and {PYTHON_MAX_VER}"
    #         )

    @validator("python_packages")
    def validate_python_packages(cls, pkgs: List[str]) -> List[str]:
        for pkg in pkgs:
            ver_parts = ()
            name, ver = pkg.split("==")

            if ver:
                ver_parts = ver.split(".")

            if not ver or len(ver_parts) <= 2:
                raise ValueError(
                    f'You must pin the package to an exact version. Got "{pkg}" expected "{name}==x.y.z"'
                )

        return pkgs

    def merged_python_pkgs(self, sep=" ") -> str:
        return sep.join(self.python_packages)

    def merged_system_pkgs(self, sep=" ") -> str:
        return sep.join(self.system_packages)

    def merged_custom_cmds(self, sep=";") -> str:
        return sep.join(self.custom_cmds)


class CustomWorkerConfig(SyftBaseModel):
    build: CustomBuildConfig
    version: str = "1"

    @classmethod
    def from_dict(cls, config: dict) -> Self:
        return cls(**config)

    @classmethod
    def from_str(cls, content: str) -> Self:
        config = yaml.safe_load(content)
        return cls.from_dict(config)

    @classmethod
    def from_path(cls, path: Path | str) -> Self:
        with open(path) as f:
            config = yaml.safe_load(f)
            return cls.from_dict(config)

    def get_signature(self) -> str:
        return sha256(self.json(sort_keys=True).encode()).hexdigest()
