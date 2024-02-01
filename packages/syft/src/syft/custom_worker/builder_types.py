# stdlib
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Optional

# third party
from pydantic import BaseModel

__all__ = ["BuilderBase", "ImageBuildResult", "ImagePushResult"]


class ImageBuildResult(BaseModel):
    image_hash: str
    logs: str


class ImagePushResult(BaseModel):
    logs: str
    exit_code: int


class BuilderBase(ABC):
    @abstractmethod
    def build_image(
        tag: str,
        dockerfile: str = None,
        dockerfile_path: Path = None,
        buildargs: Optional[dict] = None,
        **kwargs,
    ) -> ImageBuildResult:
        pass

    @abstractmethod
    def push_image(
        tag: str,
        username: str,
        password: str,
        registry_url: str,
        **kwargs,
    ) -> ImagePushResult:
        pass
