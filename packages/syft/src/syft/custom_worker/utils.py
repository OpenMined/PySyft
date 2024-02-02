# stdlib
import base64
import json
from typing import Iterable
from typing import Optional
from typing import Tuple


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


def create_dockerconfig_json(registries: Iterable[Tuple[str, str, str]]):
    config = {"auths": {}}

    for url, uname, passwd in registries:
        config["auths"][url] = {
            "username": uname,
            "password": passwd,
            "auth": base64.b64encode(f"{uname}:{passwd}".encode()).decode(),
        }

    return base64.b64encode(json.dumps(config).encode()).decode()
