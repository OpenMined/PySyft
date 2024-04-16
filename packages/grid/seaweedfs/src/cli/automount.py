# stdlib
from pathlib import Path

# third party
import yaml

# relative
from ..mount import mount_bucket
from ..mount_options import MountOptions

AUTOMOUNT_CONFIG_FILE = Path("./automount.yaml")
SUPERVISORD_CONF_DIR = Path("/etc/mounts/")
SUPERVISORD_CONF_DIR.mkdir(mode=0o600, parents=True, exist_ok=True)


def automount(config_path: Path) -> None:
    with config_path.open() as f:
        config = yaml.safe_load(f)

    mounts = config.get("mounts", [])
    for mount_opts in mounts:
        mount_opts = MountOptions(**mount_opts)
        print(
            f"Mounting type={mount_opts.remote_bucket} bucket={mount_opts.remote_bucket.bucket_name}"
        )
        mount_bucket(mount_opts, conf_dir=SUPERVISORD_CONF_DIR, overwrite=True)


if __name__ == "__main__":
    automount(config_path=AUTOMOUNT_CONFIG_FILE)
