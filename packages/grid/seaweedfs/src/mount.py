# stdlib
from hashlib import sha256
import logging
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

# relative
from .buckets import AzureCreds
from .buckets import BucketType
from .buckets import GCSCreds
from .buckets import S3Creds
from .mount_cmd import MountCmdArgs
from .mount_cmd import SupervisordConfArgs
from .mount_cmd import create_mount_cmd
from .mount_cmd import create_supervisord_conf
from .mount_cmd import create_sync_cmd
from .mount_options import MountOptions
from .util import dict_upper_keys

logger = logging.getLogger(__name__)

# This should be buckets else S3 API won't pick up the mounted volumes
LOCAL_BUCKET_PATH = "/buckets"

# Supervisord "include" path
SUPERVISORD_CONF_DIR = Path("/data/mounts/")


VALID_MOUNT_CREDENTIAL_KEYS = {
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "google_application_credentials",
    "google_project_id",
    "azure_account_name",
    "azure_account_key",
}


def mount_bucket(
    opts: MountOptions,
    conf_dir: Path = SUPERVISORD_CONF_DIR,
    overwrite: bool = False,
) -> dict:
    """Mount a remote bucket in seaweedfs"""

    # create a seaweedfs safe config name
    swfs_config_name = seaweed_safe_config_name(
        remote_name=opts.remote_bucket.type.value,
        bucket_name=opts.remote_bucket.full_bucket_name,
    )

    # check if the supervisord config already exists
    mount_conf_dir = Path(conf_dir, swfs_config_name)

    try:
        mount_conf_dir = prepare_dir(mount_conf_dir, delete_if_exist=overwrite)
    except FileExistsError:
        raise FileExistsError(f"Bucket already mounted {swfs_config_name}")

    # create the supervisord mount
    mount_env = creds_to_env(opts.remote_bucket.creds, mount_conf_dir)

    # validate args & create the mount command & run it
    mount_shell_cmd = create_mount_cmd(
        MountCmdArgs(
            config_name=swfs_config_name,
            local_bucket=opts.local_bucket,
            remote_bucket=opts.remote_bucket.bucket_name,
            remote_type=opts.remote_bucket.type.value,
            remote_creds=get_remote_cred_args(opts.remote_bucket.type),
        )
    )
    proc = subprocess.run(
        mount_shell_cmd,
        env=mount_env,
        shell=True,
        check=True,
        capture_output=True,
    )
    # weed shell will be successful, but the internal command might throw error
    if b"error" in proc.stdout or b"error" in proc.stderr:
        raise subprocess.CalledProcessError(
            proc.returncode,
            proc.args,
            output=proc.stdout,
            stderr=proc.stderr,
        )

    logger.info(f"Mount {opts.remote_bucket.bucket_name} configured.")

    # create sync command
    sync_cmd = create_sync_cmd(opts.local_bucket)

    # validate args & create the supervisord configuration & run it
    supervisord_conf_name = f"remote_mount_{swfs_config_name}"
    mount_conf_file = Path(mount_conf_dir, f"{supervisord_conf_name}.conf")
    mount_conf: str = create_supervisord_conf(
        SupervisordConfArgs(
            name=supervisord_conf_name,
            command=f'sh -c "{sync_cmd}"',
        )
    )
    # write the config to the supervisord include directory
    mount_conf_file.write_text(mount_conf)

    # update supervisord
    proc = subprocess.run(
        ["supervisorctl", "update"],
        shell=False,
        check=True,
        capture_output=True,
    )
    logger.info("Supervisor updated. stdout=%s stderr=%s", proc.stdout, proc.stderr)

    return {"name": swfs_config_name, "path": mount_conf_dir}


def seaweed_safe_config_name(remote_name: str, bucket_name: str) -> str:
    """Seaweed-friendly name for the remote config"""
    bucket_id = sha256(bucket_name.encode()).hexdigest()[:8]
    final = "mnt" + remote_name + bucket_id
    return re.sub(r"[^a-zA-Z0-9]", "", final)


def get_remote_cred_args(remote_type: BucketType) -> str:
    if remote_type == BucketType.S3:
        return "-s3.access_key=$AWS_ACCESS_KEY_ID -s3.secret_key=$AWS_SECRET_ACCESS_KEY"
    elif remote_type == BucketType.GCS:
        return "-gcs.appCredentialsFile=$GOOGLE_APPLICATION_CREDENTIALS"
    elif remote_type == BucketType.AZURE:
        return "-azure.account_name=$AZURE_ACCOUNT_NAME -azure.account_key=$AZURE_ACCOUNT_KEY"
    else:
        raise ValueError(f"Unsupported remote type: {remote_type}")


def prepare_dir(dirpath: Path, delete_if_exist: bool = True) -> Path:
    if dirpath.exists() and delete_if_exist:
        shutil.rmtree(dirpath)
    elif dirpath.exists():
        raise FileExistsError(f"Directory {dirpath} already exists")
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def creds_to_env(creds: Any, conf_dir: Path) -> dict:
    if isinstance(creds, GCSCreds):
        gcs_creds_path = creds.save(default=Path(conf_dir, "credentials.json"))
        return {
            "GOOGLE_APPLICATION_CREDENTIALS": gcs_creds_path.absolute(),
        }
    elif isinstance(creds, AzureCreds | S3Creds):
        return dict_upper_keys(creds.model_dump())
    raise ValueError("Unsupported credentials type")
