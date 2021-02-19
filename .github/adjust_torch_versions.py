# stdlib
import os
import re
import sys

VERSIONS_NONE = dict(torchvision=None, torchsprng=None)
VERSIONS_LUT = {
    "1.4.0": dict(torchvision="0.5.0", torchsprng=None),
    "1.5.0": dict(torchvision="0.6.0", torchsprng=None),
    "1.5.1": dict(torchvision="0.6.1", torchsprng=None),
    "1.6.0": dict(torchvision="0.7", torchsprng="0.1.2"),
    "1.7.0": dict(torchvision="0.8.1", torchsprng="0.1.3"),
    "1.7.1": dict(torchvision="0.8.2", torchsprng="0.1.4"),
}


def main(path_req: str, torch_version: str):
    with open(path_req, "r") as fp:
        req = fp.read()

    dep_versions = VERSIONS_LUT.get(torch_version, VERSIONS_NONE)
    dep_versions["torch"] = torch_version
    for lib in dep_versions:
        if dep_versions[lib]:
            req = re.sub(
                rf"{lib}[>=]*[\d\.]*{os.linesep}",
                f"{lib}=={dep_versions[lib]}{os.linesep}",
                req,
            )

    print(torch_version)
    with open(path_req, "w") as fp:
        fp.write(req)


if __name__ == "__main__":
    main(*sys.argv[1:])
