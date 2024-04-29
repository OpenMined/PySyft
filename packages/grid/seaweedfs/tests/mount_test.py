# stdlib
from pathlib import Path
import re
from secrets import token_hex
import shutil
from subprocess import CompletedProcess

# third party
from pytest_subprocess import FakeProcess
from pytest_subprocess.fake_popen import FakePopen

# first party
# from src.mount import create_mount_dotenv
from src.mount import mount_bucket
from src.mount import seaweed_safe_config_name
from src.mount_options import MountOptions


def test_mount_bucket_s3(fake_process: FakeProcess, random_path: Path) -> None:
    def subprocess_cb(process: FakePopen) -> CompletedProcess:
        if isinstance(process.args, str):
            assert "remote.configure" in process.args
            assert "remote.mount" in process.args
            assert "$AWS_ACCESS_KEY_ID" in process.args
            assert "$AWS_SECRET_ACCESS_KEY" in process.args
        return process

    fake_process.register(["supervisorctl", "update"], occurrences=1)
    fake_process.register([fake_process.any()], callback=subprocess_cb)

    opts = MountOptions(
        **{
            "local_bucket": token_hex(8),
            "remote_bucket": {
                "type": "s3",
                "bucket_name": token_hex(8),
                "creds": {
                    "aws_access_key_id": token_hex(8),
                    "aws_secret_access_key": token_hex(8),
                },
            },
        }
    )
    result = mount_bucket(opts, random_path)
    conf_path = result["path"]

    assert fake_process.call_count(["supervisorctl", "update"]) == 1
    assert len(list(conf_path.glob("*.conf"))) > 0
    shutil.rmtree(conf_path)


def test_mount_bucket_gcs(fake_process: FakeProcess, random_path: Path) -> None:
    def subprocess_cb(process: FakePopen) -> CompletedProcess:
        assert "remote.configure" in process.args
        assert "remote.mount" in process.args
        assert "$GOOGLE_APPLICATION_CREDENTIALS" in process.args
        return process

    fake_process.register(["supervisorctl", "update"], occurrences=1)
    fake_process.register([fake_process.any()], callback=subprocess_cb)

    opts = MountOptions(
        **{
            "local_bucket": token_hex(8),
            "remote_bucket": {
                "type": "gcs",
                "bucket_name": token_hex(8),
                "creds": {
                    "type": "service_account",
                    "project_id": token_hex(8),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "universe_domain": "googleapis.com",
                },
            },
        }
    )

    result = mount_bucket(opts, random_path)
    conf_path = result["path"]

    assert fake_process.call_count(["supervisorctl", "update"]) == 1
    assert len(list(conf_path.glob("*.conf"))) > 0
    shutil.rmtree(conf_path)


def test_mount_bucket_azure(fake_process: FakeProcess, random_path: Path) -> None:
    def subprocess_cb(process: FakePopen) -> CompletedProcess:
        assert "remote.configure" in process.args
        assert "remote.mount" in process.args
        assert "$AZURE_ACCOUNT_KEY" in process.args
        assert "$AZURE_ACCOUNT_NAME" in process.args
        return process

    fake_process.register(["supervisorctl", "update"], occurrences=1)
    fake_process.register([fake_process.any()], callback=subprocess_cb)

    opts = MountOptions(
        **{
            "local_bucket": token_hex(8),
            "remote_bucket": {
                "type": "azure",
                "container_name": token_hex(8),
                "creds": {
                    "azure_account_name": token_hex(8),
                    "azure_account_key": token_hex(8),
                },
            },
        }
    )
    result = mount_bucket(opts, random_path)
    conf_path = result["path"]

    assert fake_process.call_count(["supervisorctl", "update"]) == 1
    assert len(list(conf_path.glob("*.conf"))) > 0
    shutil.rmtree(conf_path)


TEST_REGX = r"^[a-zA-Z][a-zA-Z0-9]+$"


def test_seaweed_safe_config_name() -> None:
    a = seaweed_safe_config_name("gcs", "test-bucket-name")
    b = seaweed_safe_config_name("s3", "test_bucket_name/dir")
    c = seaweed_safe_config_name("azure", "account-name/container-name")
    d = seaweed_safe_config_name("azure", "account-name/container-name")
    e = seaweed_safe_config_name("", "")  # VALID! because of hashing

    assert a.startswith("mnt")
    assert re.match(TEST_REGX, a)

    assert b.startswith("mnt")
    assert re.match(TEST_REGX, b)

    assert c.startswith("mnt")
    assert re.match(TEST_REGX, c)

    assert c == d

    assert e.startswith("mnt")
    assert re.match(TEST_REGX, e)
