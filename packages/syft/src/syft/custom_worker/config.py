# stdlib
import contextlib
from hashlib import sha256
import io
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
import docker
from packaging import version
from pydantic import validator
from typing_extensions import Self
import yaml

# relative
from ..serde.serializable import serializable
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..types.base import SyftBaseModel
from .utils import iterator_to_string

PYTHON_DEFAULT_VER = "3.11"
PYTHON_MIN_VER = version.parse("3.10")
PYTHON_MAX_VER = version.parse("3.12")


def _malformed_python_package_error_msg(pkg: str, name: str = "package_name") -> str:
    return f'You must pin the package to an exact version. Got "{pkg}" expected "{name}==x.y.z"'


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
            name_ver = pkg.split("==")
            if len(name_ver) != 2:
                raise ValueError(_malformed_python_package_error_msg(pkg))

            name, ver = name_ver

            if ver:
                ver_parts = ver.split(".")

            if not ver or len(ver_parts) <= 2:
                raise ValueError(_malformed_python_package_error_msg(pkg, name))

        return pkgs

    def merged_python_pkgs(self, sep=" ") -> str:
        return sep.join(self.python_packages)

    def merged_system_pkgs(self, sep=" ") -> str:
        return sep.join(self.system_packages)

    def merged_custom_cmds(self, sep=";") -> str:
        return sep.join(self.custom_cmds)


class WorkerConfig(SyftBaseModel):
    pass


@serializable()
class CustomWorkerConfig(WorkerConfig):
    build: CustomBuildConfig
    version: str = "1"

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> Self:
        return cls(**config)

    @classmethod
    def from_str(cls, content: str) -> Self:
        config = yaml.safe_load(content)
        return cls.from_dict(config)

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> Self:
        with open(path) as f:
            config = yaml.safe_load(f)
            return cls.from_dict(config)

    def get_signature(self) -> str:
        return sha256(self.json(sort_keys=True).encode()).hexdigest()


@serializable()
class PrebuiltWorkerConfig(WorkerConfig):
    # tag that is already built and pushed in some registry
    tag: str
    description: Optional[str]

    def __str__(self) -> str:
        if self.description:
            return f"prebuilt tag='{self.tag}' description='{self.description}'"
        else:
            return f"prebuilt tag='{self.tag}'"

    def set_description(self, description_text: str) -> None:
        self.description = description_text


@serializable()
class DockerWorkerConfig(WorkerConfig):
    dockerfile: str
    file_name: Optional[str]
    description: Optional[str]

    @validator("dockerfile")
    def validate_dockerfile(cls, dockerfile: str) -> str:
        if not dockerfile:
            raise ValueError("Dockerfile cannot be empty")
        dockerfile = dockerfile.strip()
        return dockerfile

    @classmethod
    def from_path(
        cls,
        path: Union[Path, str],
        description: Optional[str] = "",
    ) -> Self:
        with open(path) as f:
            return cls(
                dockerfile=f.read(),
                file_name=Path(path).name,
                description=description,
            )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, DockerWorkerConfig):
            return False
        return self.dockerfile == __value.dockerfile

    def __hash__(self) -> int:
        return hash(self.dockerfile)

    def __str__(self) -> str:
        return self.dockerfile

    def set_description(self, description_text: str) -> None:
        self.description = description_text

    def test_image_build(self, tag: str, **kwargs) -> Union[SyftSuccess, SyftError]:
        try:
            with contextlib.closing(docker.from_env()) as client:
                if not client.ping():
                    return SyftError(
                        "Cannot reach docker server. Please check if docker is running."
                    )

                kwargs["fileobj"] = io.BytesIO(self.dockerfile.encode("utf-8"))
                _, logs = client.images.build(
                    tag=tag,
                    **kwargs,
                )
                return SyftSuccess(message=iterator_to_string(iterator=logs))
        except Exception as e:
            return SyftError(message=f"Failed to build: {e}")
