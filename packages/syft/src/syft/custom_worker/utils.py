# stdlib
from typing import Optional
from typing import Tuple


class ImageUtils:
    @staticmethod
    def parse_tag(tag: str) -> Tuple[Optional[str], str, str]:
        url, tag = tag.rsplit(":", 1)
        args = url.rsplit("/", 2)

        if len(args) == 3:
            registry = args[0]
            repo = "/".join(args[1:])
        else:
            registry = None
            repo = "/".join(args)

        return registry, repo, tag

    @staticmethod
    def change_registry(tag: str, registry: str) -> str:
        _, repo, tag = ImageUtils.parse_tag(tag)
        return f"{registry}/{repo}:{tag}"