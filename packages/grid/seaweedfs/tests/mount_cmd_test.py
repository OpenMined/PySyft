# stdlib
from secrets import token_hex

# third party
import pydantic
import pytest

# first party
from src.mount_cmd import MountCmdArgs
from src.mount_cmd import SupervisordConfArgs
from src.mount_cmd import create_mount_cmd
from src.mount_cmd import create_supervisord_conf


def test_mount_cmd() -> None:
    args = MountCmdArgs(
        config_name="test" + token_hex(8),
        local_bucket=token_hex(8),
        remote_bucket=token_hex(8),
        remote_type="s3",
        remote_creds="-s3.access_key=$AWS_ACCESS_KEY_ID -s3.secret_key=$AWS_SECRET",
    )
    command = create_mount_cmd(args)
    # should contain wait_for.sh
    assert "./scripts/wait_for_swfs.sh" in command
    # should configure remote
    assert f"remote.configure -name={args.config_name}" in command
    assert f"-type={args.remote_type}" in command
    # should mount remote
    assert f"remote.mount -dir=/buckets/{args.local_bucket}" in command
    assert f"-remote={args.config_name}/{args.remote_bucket}" in command


def test_mount_cmd_invalid_confname() -> None:
    with pytest.raises(pydantic.ValidationError):
        MountCmdArgs(
            # should not start with number
            config_name="1fas7891923",
            dotenv_path=f"/path/to/{token_hex(8)}.env",
            local_bucket=token_hex(8),
            remote_bucket=token_hex(8),
            remote_type="s3",
            remote_creds="-s3.access_key=$AWS_ACCESS_KEY_ID -s3.secret_key=$AWS_SECRET",
        )


def test_mount_cmd_incomplete_args() -> None:
    with pytest.raises(pydantic.ValidationError):
        MountCmdArgs(config_name="test" + token_hex(8))


def test_mount_cmd_invalid_args() -> None:
    with pytest.raises(pydantic.ValidationError):
        MountCmdArgs(
            config_name="test" + token_hex(8),
            dotenv_path=f"/path/to/{token_hex(8)}.env",
            local_bucket="",
            remote_bucket="",
            remote_type="s3",
            remote_creds="-gcs.credentials=/path/to/creds.json",
        )


def test_supervisord_conf() -> None:
    args = SupervisordConfArgs(
        name="program_" + token_hex(8),
        command="echo hello",
        priority=5,
    )
    conf = create_supervisord_conf(args)
    assert "command=echo hello" in conf
    assert "priority=5" in conf
    assert f"[program:{args.name}]"


def test_supervisord_incomplete_args() -> None:
    with pytest.raises(pydantic.ValidationError):
        SupervisordConfArgs(
            name="program_" + token_hex(8),
            priority=5,
        )
