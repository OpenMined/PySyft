# stdlib
import tarfile
from pathlib import Path

# third party
import yaml

MANIFEST_PATH = Path("./packages/syftcli/manifest.yml")
OUTPUT_PATH = Path("./build/syftcli-config/")

PREFIX_PATHS = {
    "k8s": "packages/grid/",
    "docker": "packages/grid/",
}


def create_tar(key) -> None:
    with MANIFEST_PATH.open() as fp:
        manifest = yaml.safe_load(fp)

    config_map = manifest["configFiles"]
    files = config_map[key]

    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    tarfile_path = OUTPUT_PATH / f"{key}_config.tar.gz"

    with tarfile.open(str(tarfile_path), "w:gz") as fp:
        for path in files:

            prefix = PREFIX_PATHS.get(key)
            fp.add(path, arcname=path.replace(prefix, ""))


if __name__ == "__main__":
    for config in ("docker",):
        create_tar(config)

