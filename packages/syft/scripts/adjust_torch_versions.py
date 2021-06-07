# stdlib
import platform
import re
import sys
from typing import Any
from typing import Dict

# third party
from packaging import version

VERSIONS_NONE: Dict[str, Any] = dict(torchvision=None, torchcsprng=None)
VERSIONS_LUT: Dict[str, Dict[str, Any]] = {
    "1.4.0": dict(torchvision="0.5.0", torchcsprng=None),
    "1.5.0": dict(torchvision="0.6.0", torchcsprng=None),
    "1.5.1": dict(torchvision="0.6.1", torchcsprng=None),
    "1.6.0": dict(torchvision="0.7", torchcsprng="0.1.2"),
    "1.7.0": dict(torchvision="0.8.1", torchcsprng="0.1.3"),
    "1.7.1": dict(torchvision="0.8.2", torchcsprng="0.1.4"),
    "1.8.0": dict(torchvision="0.9.0", torchcsprng="0.2.0"),
    "1.8.1": dict(torchvision="0.9.1", torchcsprng="0.2.1"),
}

system = platform.system()


def main(path_req: str, torch_version: str) -> None:
    with open(path_req, "r") as fp:
        req = fp.read()

    dep_versions = VERSIONS_LUT.get(torch_version, VERSIONS_NONE)
    dep_versions["torch"] = torch_version
    for lib in dep_versions:
        replace = ""
        if dep_versions[lib]:
            dep_version = dep_versions[lib]
            lib_version = version.parse(dep_version)
            if system != "Darwin":
                dep_version += "+cpu"  # linux and windows require +cpu
            replace = f"{lib}=={dep_version}\n"

        if (
            lib == "torchcsprng"
            and sys.version_info >= (3, 9)
            and lib_version < version.parse("0.2.0")
        ):
            replace = ""  # no torchcsprng < 0.2.0 for python 3.9

        req = re.sub(
            rf"{lib}[<>=].*\n",
            replace,
            req,
        )

    with open(path_req, "w") as fp:
        fp.write(req)


if __name__ == "__main__":
    main(*sys.argv[1:])
