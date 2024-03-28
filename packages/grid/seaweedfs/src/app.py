# stdlib
from hashlib import sha256
import json
from pathlib import Path
import subprocess

# third party
from fastapi import FastAPI
from pydantic import BaseModel

# first party
from src.mount_options import AzureBucket
from src.mount_options import GCSBucket
from src.mount_options import S3Bucket

# Supervisord "include" path
CONF_DIR = Path("/data/procs/")
CONF_DIR.mkdir(parents=True, exist_ok=True)

# this will be a part of Volume/PVC so that restarts don't lose the credentials
CREDS_DIR = Path("/data/mountapi/creds/")
CREDS_DIR.mkdir(parents=True, exist_ok=True)

# This should be buckets else S3 API won't pick up the mounted volumes
LOCAL_BUCKET_PATH = "/buckets"

app = FastAPI(title="SeaweedFS Remote Mount API")


class MountOptions(BaseModel):
    seaweed_bucket: str  # bucket to mount to
    remote: S3Bucket | GCSBucket | AzureBucket


def remote_configure_args(
    config_name: str, remote: S3Bucket | GCSBucket | AzureBucket
) -> str:
    """
    Generate the arguments for the remote.configure command

    remote.configure -name=cloud1 -type=s3 -s3.access_key=xxx -s3.secret_key=yyy -s3.storage_class="STANDARD"
    remote.configure -name=cloud2 -type=gcs -gcs.appCredentialsFile=~/service-account-file.json
    remote.configure -name=cloud3 -type=azure -azure.account_name=xxx -azure.account_key=yyy
    """

    args = [
        f"-name={config_name}",
        f"-type={remote.type}",
    ]

    if remote.type == "s3":
        args += [
            f"-s3.access_key={remote.aws_access_key}",
            f"-s3.secret_key={remote.aws_secret_key_id}",
        ]
    elif remote.type == "gcs":
        creds_path = Path(CREDS_DIR, f"{config_name}.json")
        creds_path.write_text(json.dumps(remote.gcs_credentials))
        args += [
            f"-gcs.appCredentialsFile={str(creds_path)}",
        ]
    elif remote.type == "azure":
        args += [
            f"-azure.account_name={remote.azure_account_name}",
            f"-azure.account_key={remote.azure_account_key}",
        ]

    return " ".join(args)


def remote_mount_args(filer_dir: str, remote_path: str) -> str:
    """
    Generate the arguments for the remote.mount command

    remote.mount -dir=/buckets/bucket-name -remote=cloud1/bucket-name
    """

    args = [
        f"-dir={filer_dir}",
        f"-remote={remote_path}",
    ]

    return " ".join(args)


def seaweed_config_name(prefix: str, bucket_name: str) -> str:
    """Seaweed-friendly name for the remote config"""
    return prefix + sha256(bucket_name.encode()).hexdigest()[:8]


@app.get("/")
def ping() -> dict:
    return {"success": True}


def create_sync_conf(
    name: str,
    config_args: str,
    mount_args: str,
    filer_dir: str,
) -> str:
    """Generate the supervisord configuration for the sync process"""

    args = [
        f"echo remote.configure {config_args} | weed shell",
        f"echo remote.mount {mount_args} | weed shell",
        f"weed filer.remote.sync -dir={filer_dir}",
    ]
    chained_args = " && ".join(args)

    command = f'bash -c "{chained_args}"'

    return (
        f"[program:remote_sync_{name}]\n"
        f"command={command}\n"
        f"autorestart=unexpected\n"
        f"exitcodes=0\n"
        f"stdout_logfile=/dev/stdout\n"
        f"stdout_logfile_maxbytes=0\n"
        f"redirect_stderr=true\n"
    )


@app.post("/mount/")
def mount(opts: MountOptions) -> dict:
    config_name = seaweed_config_name(opts.remote.type, opts.remote.bucket_name)
    filer_dir = f"{LOCAL_BUCKET_PATH}/{opts.remote.bucket_name}"
    remote_path = f"{config_name}/{opts.remote.bucket_name}"

    # a file we save to check if the bucket is already mounted
    # without having to query the seaweed master through CLI
    config_sync_path = Path(CONF_DIR, f"{config_name}.conf")

    if config_sync_path.exists():
        return {
            "success": False,
            "message": f"Bucket {opts.seaweed_bucket} already mounted",
        }

    config_args: str = remote_configure_args(config_name, opts.remote)
    mount_args: str = remote_mount_args(filer_dir, remote_path)

    # create a supervisord config for the mount+sync process
    sync_conf = create_sync_conf(config_name, config_args, mount_args, filer_dir)
    config_sync_path.write_text(sync_conf)

    # update supervisord
    res = subprocess.run(["supervisorctl", "update"])

    # TODO: query supervisord process status to check if the process is running
    return {
        "remote_config": config_args,
        "remote_mount": mount_args,
        "sync_conf": sync_conf,
        "ctl_return": res.returncode,
    }


@app.post("/configure_azure")
def configure_azure(first_res: dict) -> str:
    account_name = first_res["account_name"]
    account_key = first_res["account_key"]
    container_name = first_res["container_name"]
    remote_name = first_res["remote_name"]
    bucket_name = first_res["bucket_name"]

    res = subprocess.run(
        [
            "bash",
            "scripts/mount_command.sh",
            remote_name,
            account_name,
            bucket_name,
            container_name,
            account_key,
        ]
    )
    return str(res.returncode)
