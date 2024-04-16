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
from pymongo import MongoClient
import pytest

# syft absolute
import syft as sy
from syft.abstract_node import NodeSideType
from syft.client.domain_client import DomainClient
from syft.node.worker import Worker
from syft.protocol.data_protocol import get_data_protocol
from syft.protocol.data_protocol import protocol_release_dir
from syft.protocol.data_protocol import stage_protocol_changes
from syft.service.user import user

# relative
from .syft.stores.store_fixtures_test import dict_action_store  # noqa: F401
from .syft.stores.store_fixtures_test import dict_document_store  # noqa: F401
from .syft.stores.store_fixtures_test import dict_queue_stash  # noqa: F401
from .syft.stores.store_fixtures_test import dict_store_partition  # noqa: F401
from .syft.stores.store_fixtures_test import mongo_action_store  # noqa: F401
from .syft.stores.store_fixtures_test import mongo_document_store  # noqa: F401
from .syft.stores.store_fixtures_test import mongo_queue_stash  # noqa: F401
from .syft.stores.store_fixtures_test import mongo_store_partition  # noqa: F401
from .syft.stores.store_fixtures_test import sqlite_action_store  # noqa: F401
from .syft.stores.store_fixtures_test import sqlite_document_store  # noqa: F401
from .syft.stores.store_fixtures_test import sqlite_queue_stash  # noqa: F401
from .syft.stores.store_fixtures_test import sqlite_store_partition  # noqa: F401
from .syft.stores.store_fixtures_test import sqlite_workspace  # noqa: F401
from .utils.mongodb import start_mongo_server
from .utils.mongodb import stop_mongo_server
from .utils.xdist_state import SharedState


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
        for _file_path in protocol_release_dir().iterdir():
            for version in dp.read_json(_file_path):
                if version not in dp.protocol_history.keys():
                    _file_path.unlink()


@pytest.fixture
def faker():
    yield Faker()


@pytest.fixture(scope="function")
def worker() -> Worker:
    worker = sy.Worker.named(name=token_hex(8))
    yield worker
    worker.cleanup()
    del worker


@pytest.fixture(scope="function")
def second_worker() -> Worker:
    # Used in node syncing tests
    worker = sy.Worker.named(name=token_hex(8))
    yield worker
    worker.cleanup()
    del worker


@pytest.fixture(scope="function")
def high_worker() -> Worker:
    worker = sy.Worker.named(name=token_hex(8), node_side_type=NodeSideType.HIGH_SIDE)
    yield worker
    worker.cleanup()
    del worker


@pytest.fixture(scope="function")
def low_worker() -> Worker:
    worker = sy.Worker.named(name=token_hex(8), node_side_type=NodeSideType.LOW_SIDE)
    yield worker
    worker.cleanup()
    del worker


@pytest.fixture(scope="function")
def full_high_worker(n_consumers: int = 3, create_producer: bool = True) -> Worker:
    _node = sy.orchestra.launch(
        node_side_type=NodeSideType.HIGH_SIDE,
        name=token_hex(8),
        # dev_mode=True,
        reset=True,
        n_consumers=n_consumers,
        create_producer=create_producer,
        queue_port=None,
        in_memory_workers=True,
        local_db=False,
        thread_workers=False,
    )
    # startup code here
    yield _node
    # Cleanup code
    _node.python_node.cleanup()
    _node.land()


@pytest.fixture(scope="function")
def full_low_worker(n_consumers: int = 3, create_producer: bool = True) -> Worker:
    _node = sy.orchestra.launch(
        node_side_type=NodeSideType.LOW_SIDE,
        name=token_hex(8),
        # dev_mode=True,
        reset=True,
        n_consumers=n_consumers,
        create_producer=create_producer,
        queue_port=None,
        in_memory_workers=True,
        local_db=False,
        thread_workers=False,
    )
    # startup code here
    yield _node
    # # Cleanup code
    _node.python_node.cleanup()
    _node.land()


@pytest.fixture
def root_domain_client(worker) -> DomainClient:
    yield worker.root_client


@pytest.fixture
def root_verify_key(worker):
    yield worker.root_client.credentials.verify_key


@pytest.fixture
def guest_client(worker) -> DomainClient:
    yield worker.guest_client


@pytest.fixture
def guest_verify_key(worker):
    yield worker.guest_client.credentials.verify_key


@pytest.fixture
def guest_domain_client(root_domain_client) -> DomainClient:
    yield root_domain_client.guest()


@pytest.fixture
def document_store(worker):
    yield worker.document_store
    worker.document_store.reset()


@pytest.fixture
def action_store(worker):
    yield worker.action_store


@pytest.fixture(scope="session")
def mongo_client(testrun_uid):
    """
    A race-free fixture that starts a MongoDB server for an entire pytest session.
    Cleans up the server when the session ends, or when the last client disconnects.
    """
    db_name = f"pytest_mongo_{testrun_uid}"
    root_dir = Path(gettempdir(), db_name)
    state = SharedState(db_name)
    KEY_CONN_STR = "mongoConnectionString"
    KEY_CLIENTS = "mongoClients"

    # start the server if it's not already running
    with state.lock:
        conn_str = state.get(KEY_CONN_STR, None)

        if not conn_str:
            conn_str = start_mongo_server(db_name)
            state.set(KEY_CONN_STR, conn_str)

        # increment the number of clients
        clients = state.get(KEY_CLIENTS, 0) + 1
        state.set(KEY_CLIENTS, clients)

    # create a client, and test the connection
    client = MongoClient(conn_str)
    assert client.server_info().get("ok") == 1.0

    yield client

    # decrement the number of clients
    with state.lock:
        clients = state.get(KEY_CLIENTS, 0) - 1
        state.set(KEY_CLIENTS, clients)

    # if no clients are connected, destroy the server
    if clients <= 0:
        stop_mongo_server(db_name)
        state.purge()
        shutil.rmtree(root_dir, ignore_errors=True)


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
    "tests.syft.action_graph.fixtures",
    "tests.syft.serde.fixtures",
]
