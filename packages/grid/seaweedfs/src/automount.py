# stdlib
import logging
import logging.config
from pathlib import Path
import subprocess

# third party
import yaml

# relative
from .mount import mount_bucket
from .mount_options import MountOptions

logger = logging.getLogger(__name__)


def automount(automount_conf: Path, mount_conf_dir: Path) -> None:
    with automount_conf.open() as f:
        config = yaml.safe_load(f)

    mounts = config.get("mounts", [])
    for mount_opts in mounts:
        mount_opts = MountOptions(**mount_opts)
        try:
            logger.info(
                f"Auto mount type={mount_opts.remote_bucket.type} "
                f"bucket={mount_opts.remote_bucket.bucket_name}"
            )
            result = mount_bucket(
                mount_opts,
                conf_dir=mount_conf_dir,
                overwrite=True,
            )
            logger.info(f"Auto mount success. {result}")
        except FileExistsError as e:
            logger.info(e)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Mount error: code={e.returncode} stdout={e.stdout} stderr={e.stderr}"
            )
        except Exception as e:
            logger.error(f"Unhandled exception: {e}")
