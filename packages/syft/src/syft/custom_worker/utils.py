# stdlib
from collections.abc import Iterable
import json


def iterator_to_string(iterator: Iterable) -> str:
    log = ""
    for line in iterator:
        for item in line.values():
            if isinstance(item, str):
                log += item
            elif isinstance(item, dict):
                log += json.dumps(item) + "\n"
            else:
                log += str(item)
    return log


class ImageUtils:
    @staticmethod
    def parse_tag(tag: str) -> tuple[str | None, str, str]:
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
