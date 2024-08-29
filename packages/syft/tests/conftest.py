# stdlib
from functools import cache
import os
from pathlib import Path
from secrets import token_hex
import shutil
import sys
from tempfile import gettempdir
from unittest import mock

# third party
from faker import Faker
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft import Dataset
from syft.abstract_server import ServerSideType
from syft.client.datasite_client import DatasiteClient
from syft.protocol.data_protocol import get_data_protocol
from syft.protocol.data_protocol import protocol_release_dir
from syft.protocol.data_protocol import stage_protocol_changes
from syft.server.worker import Worker
from syft.service.user import user

# relative
# our version of mongomock that has a fix for CodecOptions and custom TypeRegistry Support
from .mongomock.mongo_client import MongoClient
from .syft.stores.store_fixtures_test import dict_action_store
from .syft.stores.store_fixtures_test import dict_document_store
from .syft.stores.store_fixtures_test import dict_queue_stash
from .syft.stores.store_fixtures_test import dict_store_partition
from .syft.stores.store_fixtures_test import mongo_action_store
from .syft.stores.store_fixtures_test import mongo_document_store
from .syft.stores.store_fixtures_test import mongo_queue_stash
from .syft.stores.store_fixtures_test import mongo_store_partition
from .syft.stores.store_fixtures_test import sqlite_action_store
from .syft.stores.store_fixtures_test import sqlite_document_store
from .syft.stores.store_fixtures_test import sqlite_queue_stash
from .syft.stores.store_fixtures_test import sqlite_store_partition
from .syft.stores.store_fixtures_test import sqlite_workspace


def patch_protocol_file(filepath: Path):
    dp = get_data_protocol()
    shutil.copyfile(src=dp.file_path, dst=filepath)


def remove_file(filepath: Path):
    filepath.unlink(missing_ok=True)


def pytest_sessionstart(session):
    # add env var SYFT_TEMP_ROOT to create a unique temp dir for each test run
    os.environ["SYFT_TEMP_ROOT"] = f"pytest_syft_{token_hex(8)}"


def pytest_configure(config):
    if hasattr(config, "workerinput") or is_vscode_discover():
        return

    for path in Path(gettempdir()).glob("pytest_*"):
        shutil.rmtree(path, ignore_errors=True)

    for path in Path(gettempdir()).glob("sherlock"):
        shutil.rmtree(path, ignore_errors=True)


def is_vscode_discover():
    """Check if the test is being run from VSCode discover test runner."""

    cmd = " ".join(sys.argv)
    return "ms-python.python" in cmd and "discover" in cmd


# Pytest hook to set the number of workers for xdist
def pytest_xdist_auto_num_workers(config):
    num = config.option.numprocesses
    if num == "auto" or num == "logical":
        return os.cpu_count()
    return None


def pytest_collection_modifyitems(items):
    for item in items:
        item_fixtures = getattr(item, "fixturenames", ())
        if "sqlite_workspace" in item_fixtures:
            item.add_marker(pytest.mark.xdist_group(name="sqlite"))


@pytest.fixture(autouse=True)
def protocol_file():
    random_name = sy.UID().to_string()
    protocol_dir = sy.SYFT_PATH / "protocol"
    file_path = protocol_dir / f"{random_name}.json"
    patch_protocol_file(filepath=file_path)
    try:
        yield file_path
    finally:
        remove_file(file_path)


@pytest.fixture(autouse=True)
def stage_protocol(protocol_file: Path):
    with mock.patch(
        "syft.protocol.data_protocol.PROTOCOL_STATE_FILENAME",
        protocol_file.name,
    ):
        dp = get_data_protocol()
        stage_protocol_changes()
        # bump_protocol_version()
        yield dp.protocol_history
        dp.reset_dev_protocol()
        dp.save_history(dp.protocol_history)

        # Cleanup release dir, remove unused released files
        if os.path.exists(protocol_release_dir()):
            for _file_path in protocol_release_dir().iterdir():
                for version in dp.read_json(_file_path):
                    if version not in dp.protocol_history.keys():
                        _file_path.unlink()


@pytest.fixture()
def faker():
    return Faker()


@pytest.fixture(scope="function")
def worker() -> Worker:
    worker = sy.Worker.named(name=token_hex(8))
    yield worker
    worker.cleanup()
    del worker


@pytest.fixture(scope="function")
def second_worker() -> Worker:
    # Used in server syncing tests
    worker = sy.Worker.named(name=token_hex(8))
    yield worker
    worker.cleanup()
    del worker


@pytest.fixture(scope="function")
def high_worker() -> Worker:
    worker = sy.Worker.named(
        name=token_hex(8), server_side_type=ServerSideType.HIGH_SIDE
    )
    yield worker
    worker.cleanup()
    del worker


@pytest.fixture(scope="function")
def low_worker() -> Worker:
    worker = sy.Worker.named(
        name=token_hex(8), server_side_type=ServerSideType.LOW_SIDE, dev_mode=True
    )
    yield worker
    worker.cleanup()
    del worker


@pytest.fixture()
def root_datasite_client(worker) -> DatasiteClient:
    return worker.root_client


@pytest.fixture()
def root_verify_key(worker):
    return worker.root_client.credentials.verify_key


@pytest.fixture()
def guest_client(worker) -> DatasiteClient:
    return worker.guest_client


@pytest.fixture()
def guest_verify_key(worker):
    return worker.guest_client.credentials.verify_key


@pytest.fixture()
def guest_datasite_client(root_datasite_client) -> DatasiteClient:
    return root_datasite_client.guest()


@pytest.fixture()
def ds_client(
    faker: Faker, root_datasite_client: DatasiteClient, guest_client: DatasiteClient
):
    guest_email = faker.email()
    password = "mysecretpassword"
    root_datasite_client.register(
        name=faker.name(),
        email=guest_email,
        password=password,
        password_verify=password,
    )
    ds_client = guest_client.login(email=guest_email, password=password)
    yield ds_client


@pytest.fixture()
def ds_verify_key(ds_client: DatasiteClient):
    yield ds_client.credentials.verify_key


@pytest.fixture()
def document_store(worker):
    yield worker.document_store
    worker.document_store.reset()


@pytest.fixture()
def action_store(worker):
    return worker.action_store


@pytest.fixture(scope="session")
def mongo_client(testrun_uid):
    """
    A race-free fixture that starts a MongoDB server for an entire pytest session.
    Cleans up the server when the session ends, or when the last client disconnects.
    """
    db_name = f"pytest_mongo_{testrun_uid}"

    # rand conn str
    conn_str = f"mongodb://localhost:27017/{db_name}"

    # create a client, and test the connection
    client = MongoClient(conn_str)
    assert client.server_info().get("ok") == 1.0

    return client

    # stop_mongo_server(db_name)


@pytest.fixture(autouse=True)
def patched_mongo_client(monkeypatch):
    monkeypatch.setattr("pymongo.mongo_client.MongoClient", MongoClient)


@pytest.fixture(autouse=True)
def patched_session_cache(monkeypatch):
    # patching compute heavy hashing to speed up tests

    def _get_key(email, password, connection):
        return f"{email}{password}{connection}"

    monkeypatch.setattr("syft.client.client.SyftClientSessionCache._get_key", _get_key)


cached_salt_and_hash_password = cache(user.salt_and_hash_password)
cached_check_pwd = cache(user.check_pwd)


@pytest.fixture(autouse=True)
def patched_user(monkeypatch):
    # patching compute heavy hashing to speed up tests

    monkeypatch.setattr(
        "syft.service.user.user.salt_and_hash_password",
        cached_salt_and_hash_password,
    )
    monkeypatch.setattr(
        "syft.service.user.user.check_pwd",
        cached_check_pwd,
    )


@pytest.fixture
def small_dataset() -> Dataset:
    dataset = Dataset(
        name="small_dataset",
        asset_list=[
            sy.Asset(
                name="small_dataset",
                data=np.array([1, 2, 3]),
                mock=np.array([1, 1, 1]),
            )
        ],
    )
    yield dataset


@pytest.fixture
def big_dataset() -> Dataset:
    num_elements = 20 * 1024 * 1024
    data_big = np.random.randint(0, 100, size=num_elements)
    mock_big = np.random.randint(0, 100, size=num_elements)
    dataset = Dataset(
        name="big_dataset",
        asset_list=[
            sy.Asset(
                name="big_dataset",
                data=data_big,
                mock=mock_big,
            )
        ],
    )
    yield dataset


__all__ = [
    "mongo_store_partition",
    "mongo_document_store",
    "mongo_queue_stash",
    "mongo_action_store",
    "sqlite_store_partition",
    "sqlite_workspace",
    "sqlite_document_store",
    "sqlite_queue_stash",
    "sqlite_action_store",
    "dict_store_partition",
    "dict_action_store",
    "dict_document_store",
    "dict_queue_stash",
]

pytest_plugins = [
    "tests.syft.users.fixtures",
    "tests.syft.settings.fixtures",
    "tests.syft.request.fixtures",
    "tests.syft.dataset.fixtures",
    "tests.syft.notifications.fixtures",
    "tests.syft.serde.fixtures",
]
