# stdlib
from collections.abc import Iterable
import contextlib
import io
from pathlib import Path
from typing import Any

# third party
import docker

# relative
from .builder_types import BUILD_IMAGE_TIMEOUT_SEC
from .builder_types import BuilderBase
from .builder_types import ImageBuildResult
from .builder_types import ImagePushResult
from .utils import iterator_to_string

__all__ = ["DockerBuilder"]


class DockerBuilder(BuilderBase):
    def build_image(
        self,
        tag: str,
        dockerfile: str | None = None,
        dockerfile_path: Path | None = None,
        buildargs: dict | None = None,
        **kwargs: Any,
    ) -> ImageBuildResult:
        if dockerfile:
            # convert dockerfile string to file-like object
            kwargs["fileobj"] = io.BytesIO(dockerfile.encode("utf-8"))
        elif dockerfile_path:
            # context dir + dockerfile name
            kwargs["path"] = str(dockerfile_path.parent)
            kwargs["dockerfile"] = str(dockerfile_path.name)

        # Core docker build call. Func kwargs should match with Docker SDK's BuildApiMixin
        with contextlib.closing(docker.from_env()) as client:
            image_result, logs = client.images.build(
                tag=tag,
                timeout=BUILD_IMAGE_TIMEOUT_SEC,
                buildargs=buildargs,
                rm=True,
                labels={"orgs.openmined.syft": f"Build image {tag}"},
                **kwargs,
            )
            return ImageBuildResult(
                # An image that is built locally will not have a RepoDigests until pushed to a v2 registry
                # https://stackoverflow.com/a/39812035
                image_hash=image_result.id,
                logs=self._parse_output(logs),
            )

    def push_image(
        self,
        tag: str,
        registry_url: str,
        username: str | None = None,
        password: str | None = None,
        **kwargs: Any,
    ) -> ImagePushResult:
        with contextlib.closing(docker.from_env()) as client:
            if registry_url and username and password:
                client.login(
                    username=username,
                    password=password,
                    registry=registry_url,
                )

            result = client.images.push(repository=tag)
            return ImagePushResult(logs=result, exit_code=0)

    def _parse_output(self, log_iterator: Iterable) -> str:
        return iterator_to_string(iterator=log_iterator)
