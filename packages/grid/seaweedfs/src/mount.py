# stdlib
import json
from pathlib import Path

# relative
from .mount_options import MountOptions
from .util import seaweed_config_name

# This should be buckets else S3 API won't pick up the mounted volumes
LOCAL_BUCKET_PATH = "/buckets"

# Supervisord "include" path
MOUNT_CONF_DIR = Path("/data/mount/procs/")
MOUNT_CONF_DIR.mkdir(parents=True, exist_ok=True)

# this will be a part of Volume/PVC so that restarts don't lose the credentials
MOUNT_CREDS_DIR = Path("/data/mount/creds/")
MOUNT_CREDS_DIR.mkdir(parents=True, exist_ok=True)


def mount_bucket(opts: MountOptions, conf_dir: Path = MOUNT_CONF_DIR) -> dict:
    """Mount a remote bucket in seaweedfs"""

    # create a seaweed-valid unique name for the config
    config_name = seaweed_config_name(
        prefix=opts.remote_bucket.type,
        bucket_name=opts.remote_bucket.bucket_name,
    )

    # check if the supervisord config already exists
    mount_conf_path = Path(conf_dir, f"{config_name}.conf")

    if mount_conf_path.exists():
        raise FileExistsError(f"Bucket {opts.local_bucket} already mounted")

    # create the supervisord mount config
    mount_conf = create_mount_conf(config_name, opts)

    # write the config to the supervisord include directory
    mount_conf_path.write_text(mount_conf)

    return {
        "config_name": config_name,
        "mount_conf_path": mount_conf_path,
    }


def create_mount_conf(config_name: str, opts: MountOptions) -> str:
    """Generate the supervisord configuration for mounting a remote bucket"""

    sw_filer_dir = f"{LOCAL_BUCKET_PATH}/{opts.remote_bucket.bucket_name}"
    sw_remote_path = f"{config_name}/{opts.remote_bucket.bucket_name}"
    sw_remote_type = opts.remote_bucket.type

    # save the credentials to a file for it to be available during restarts
    creds_path = save_creds(config_name, opts.remote_bucket.creds)
    if not creds_path.exists():
        raise FileNotFoundError(f"Credentials not available: {creds_path}")

    # setup the commands to configure and mount the remote bucket
    sw_config_cmd: str = remote_configure_cmd(config_name, sw_remote_type, creds_path)
    sw_mount_cmd: str = remote_mount_cmd(sw_filer_dir, sw_remote_path)
    sw_sync_cmd: str = filer_sync_cmd(sw_filer_dir)
    sw_wait_cmd: str = "sh ./scripts/wait_for_swfs.sh"

    # mount is a combination of configure, mount and sync
    mount_cmd: str = run_cmd(
        [
            sw_wait_cmd,
            sw_config_cmd,
            sw_mount_cmd,
            sw_sync_cmd,
        ]
    )

    # create the supervisord configuration
    supervisord_conf_name = f"remote_mount_{config_name}"
    conf: str = create_supervisord_conf(supervisord_conf_name, mount_cmd)
    return conf


def save_creds(config_name: str, creds: Path | dict) -> Path:
    """Save the credentials to a file and return the path"""

    if isinstance(creds, Path):
        return creds

    creds_path = Path(MOUNT_CREDS_DIR, f"{config_name}.json")
    creds_path.write_text(json.dumps(creds))
    return creds_path


def remote_configure_cmd(
    config_name: str,
    remote_type: str,
    creds_path: str | Path,
) -> str:
    """
    Generate the arguments for the remote.configure command

    remote.configure -name=cloud1 -type=s3 -s3.access_key=xxx -s3.secret_key=yyy -s3.storage_class="STANDARD"
    remote.configure -name=cloud2 -type=gcs -gcs.appCredentialsFile=~/service-account-file.json
    remote.configure -name=cloud3 -type=azure -azure.account_name=xxx -azure.account_key=yyy
    """

    args = [
        f"-name={config_name}",
        f"-type={remote_type}",
        # load key lazily using a get_secret script to not expose it in plain text
        f"$(python -m src.cli.remote_config_secret {remote_type} {creds_path})",
    ]

    str_args = " ".join(args)
    return f"echo remote.configure {str_args} | weed shell"


def remote_mount_cmd(filer_dir: str, remote_path: str) -> str:
    """
    Generate the arguments for the remote.mount command

    remote.mount -dir=/buckets/bucket-name -remote=cloud1/bucket-name
    """

    args = [
        f"-dir={filer_dir}",
        f"-remote={remote_path}",
    ]

    str_args = " ".join(args)
    return f"echo remote.mount {str_args} | weed shell"


def filer_sync_cmd(filer_dir: str) -> str:
    """
    Generate the arguments for the filer.remote.sync command

    weed filer.remote.sync -dir=/buckets/bucket-name
    """

    return f"weed filer.remote.sync -dir={filer_dir}"


def run_cmd(cmds: list[str]) -> str:
    """Generate the command to mount the remote bucket"""

    chained_args = " && ".join(cmds)
    return f'sh -ec "{chained_args}"'


def create_supervisord_conf(
    name: str,
    command: str,
    priority: int = 5,
) -> str:
    """Generate the supervisord configuration for a command"""

    return (
        f"[program:{name}]\n"
        f"priority={priority}\n"
        f"command={command}\n"
        f"stdout_logfile=/dev/stdout\n"
        f"stdout_logfile_maxbytes=0\n"
        f"redirect_stderr=true\n"
    )
