# stdlib
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

# third party
from pydantic import BaseModel

__all__ = [
    "BuilderBase",
    "ImageBuildResult",
    "ImagePushResult",
    "BUILD_IMAGE_TIMEOUT_SEC",
    "PUSH_IMAGE_TIMEOUT_SEC",
]


BUILD_IMAGE_TIMEOUT_SEC = 30 * 60
PUSH_IMAGE_TIMEOUT_SEC = 10 * 60


class ImageBuildResult(BaseModel):
    image_hash: str
    logs: str


class ImagePushResult(BaseModel):
    logs: str
    exit_code: int

    @property
    def has_failed(self) -> bool:
        return self.exit_code != 0


class BuilderBase(ABC):
    @abstractmethod
    def build_image(
        self,
        tag: str,
        dockerfile: str | None = None,
        dockerfile_path: Path | None = None,
        buildargs: dict | None = None,
        **kwargs: Any,
    ) -> ImageBuildResult:
        pass

    @abstractmethod
    def push_image(
        self,
        tag: str,
        registry_url: str,
        username: str | None = None,
        password: str | None = None,
        **kwargs: Any,
    ) -> ImagePushResult:
        pass
