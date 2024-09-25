# stdlib
import contextlib
from hashlib import sha256
import io
from pathlib import Path
from typing import Any

# third party
import docker
from packaging import version
from pydantic import field_validator
from typing_extensions import Self
import yaml

# relative
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
from ..service.response import SyftSuccess
from ..types.base import SyftBaseModel
from ..types.errors import SyftException
from .utils import iterator_to_string

PYTHON_DEFAULT_VER = "3.12"
PYTHON_MIN_VER = version.parse("3.10")
PYTHON_MAX_VER = version.parse("3.12")


def _malformed_python_package_error_msg(pkg: str, name: str = "package_name") -> str:
    return f'You must pin the package to an exact version. Got "{pkg}" expected "{name}==x.y.z"'


class CustomBuildConfig(SyftBaseModel):
    gpu: bool = False
    # python_version: str = PYTHON_DEFAULT_VER
    python_packages: list[str] = []
    system_packages: list[str] = []
    custom_cmds: list[str] = []

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

    @field_validator("python_packages")
    @classmethod
    def validate_python_packages(cls, pkgs: list[str]) -> list[str]:
        for pkg in pkgs:
            ver_parts: tuple | list = ()
            name_ver = pkg.split("==")
            if len(name_ver) != 2:
                raise ValueError(_malformed_python_package_error_msg(pkg))

            name, ver = name_ver

            if ver:
                ver_parts = ver.split(".")

            if not ver or len(ver_parts) <= 2:
                raise ValueError(_malformed_python_package_error_msg(pkg, name))

        return pkgs

    def merged_python_pkgs(self, sep: str = " ") -> str:
        return sep.join(self.python_packages)

    def merged_system_pkgs(self, sep: str = " ") -> str:
        return sep.join(self.system_packages)

    def merged_custom_cmds(self, sep: str = ";") -> str:
        return sep.join(self.custom_cmds)


@serializable(canonical_name="WorkerConfig", version=1)
class WorkerConfig(SyftBaseModel):
    pass

    def hash(self) -> str:
        _bytes = _serialize(self, to_bytes=True, for_hashing=True)
        return sha256(_bytes).digest().hex()


@serializable(canonical_name="CustomWorkerConfig", version=1)
class CustomWorkerConfig(WorkerConfig):
    build: CustomBuildConfig
    version: str = "1"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
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


@serializable(canonical_name="PrebuiltWorkerConfig", version=1)
class PrebuiltWorkerConfig(WorkerConfig):
    # tag that is already built and pushed in some registry
    tag: str
    description: str | None = None

    def __str__(self) -> str:
        if self.description:
            return f"prebuilt tag='{self.tag}' description='{self.description}'"
        else:
            return f"prebuilt tag='{self.tag}'"

    def set_description(self, description_text: str) -> None:
        self.description = description_text

    def __hash__(self) -> int:
        return hash(self.tag)


@serializable(canonical_name="DockerWorkerConfig", version=1)
class DockerWorkerConfig(WorkerConfig):
    dockerfile: str
    file_name: str | None = None
    description: str | None = None

    @field_validator("dockerfile")
    @classmethod
    def validate_dockerfile(cls, dockerfile: str) -> str:
        if not dockerfile:
            raise ValueError("Dockerfile cannot be empty")
        dockerfile = dockerfile.strip()
        return dockerfile

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        description: str | None = "",
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

    def test_image_build(self, tag: str, **kwargs: Any) -> SyftSuccess:
        try:
            with contextlib.closing(docker.from_env()) as client:
                if not client.ping():
                    raise SyftException(
                        "Cannot reach docker server. Please check if docker is running."
                    )

                kwargs["fileobj"] = io.BytesIO(self.dockerfile.encode("utf-8"))
                _, logs = client.images.build(
                    tag=tag,
                    rm=True,
                    labels={"orgs.openmined.syft": "Test image build"},
                    **kwargs,
                )
                return SyftSuccess(message=iterator_to_string(iterator=logs))
        except Exception as e:
            # stdlib
            import traceback

            raise SyftException(
                public_message=f"Failed to build: {e} {traceback.format_exc()}"
            )
