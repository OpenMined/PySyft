# stdlib

# third party
from pydantic import BaseModel
from pydantic import Field

__MOUNT_CMD_TEMPLATE = """
. ./scripts/wait_for_swfs.sh &&
echo remote.configure -name={config_name} -type={remote_type} {remote_creds} | weed shell &&
echo remote.mount -dir=/buckets/{local_bucket} -remote={config_name}/{remote_bucket} | weed shell
"""

__SYNC_CMD_TEMPLATE = """
. ./scripts/wait_for_swfs.sh &&
exec weed filer.remote.sync -dir=/buckets/{local_bucket}
"""

__SUPERVISORD_TEMPLATE = """
[program:{name}]
priority={priority}
command={command}
autostart=true
autorestart=unexpected
exitcodes=0
startretries=0
startsecs=0
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
"""


class MountCmdArgs(BaseModel):
    config_name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9]*$")
    local_bucket: str = Field(min_length=1)
    remote_bucket: str = Field(min_length=1)
    remote_type: str = Field(min_length=1)
    remote_creds: str = Field(min_length=1, pattern=r"-(s3|gcs|azure)\.\w+")


class SupervisordConfArgs(BaseModel):
    name: str = Field(min_length=1)
    command: str = Field(min_length=1)
    priority: int = Field(default=99, gt=1)


def create_mount_cmd(args: MountCmdArgs) -> str:
    """Generate the seaweedfs mount command"""

    return __MOUNT_CMD_TEMPLATE.format(**args.model_dump()).replace("\n", " ").strip()


def create_sync_cmd(local_bucket: str) -> str:
    args = {"local_bucket": local_bucket}
    return __SYNC_CMD_TEMPLATE.format(**args).replace("\n", " ").strip()


def create_supervisord_conf(args: SupervisordConfArgs) -> str:
    """Generate the supervisord configuration for a command"""

    return __SUPERVISORD_TEMPLATE.format(**args.model_dump()).strip()
