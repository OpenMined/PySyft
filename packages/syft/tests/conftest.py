# stdlib
import os
from pathlib import Path
import shutil
from unittest import mock

# third party
from faker import Faker
from pymongo import MongoClient
import pytest

# syft absolute
import syft as sy
from syft.client.client import SyftClientSessionCache
from syft.client.domain_client import DomainClient
from syft.node.worker import Worker
from syft.protocol.data_protocol import get_data_protocol
from syft.protocol.data_protocol import protocol_release_dir
from syft.protocol.data_protocol import stage_protocol_changes

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


# Pytest hook to set the number of workers for xdist
def pytest_xdist_auto_num_workers(config):
    num = config.option.numprocesses
    if num == "auto" or num == "logical":
        return os.cpu_count()
    return None


# def pytest_collection_modifyitems(items):
#     for item in items:
#         item_fixtures = getattr(item, "fixturenames", ())
#         if "test_sqlite_" in item.nodeid:
#             item.add_marker(pytest.mark.xdist_group(name="sqlite"))


@pytest.fixture(autouse=True)
def protocol_file():
    random_name = sy.UID().to_string()
    protocol_dir = sy.SYFT_PATH / "protocol"
    file_path = protocol_dir / f"{random_name}.json"
    patch_protocol_file(filepath=file_path)
    yield file_path
    remove_file(filepath=file_path)


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
        dp.revert_latest_protocol()
        dp.save_history(dp.protocol_history)

        # Cleanup release dir, remove unused released files
        for _file_path in protocol_release_dir().iterdir():
            for version in dp.read_json(_file_path):
                if version not in dp.protocol_history.keys():
                    _file_path.unlink()


@pytest.fixture()
def faker():
    return Faker()


@pytest.fixture()
def worker(faker) -> Worker:
    worker = sy.Worker.named(name=faker.name())
    yield worker
    worker.stop()
    del worker


@pytest.fixture()
def root_domain_client(worker) -> DomainClient:
    return worker.root_client


@pytest.fixture()
def root_verify_key(worker):
    return worker.root_client.credentials.verify_key


@pytest.fixture()
def guest_client(worker) -> DomainClient:
    return worker.guest_client


@pytest.fixture()
def guest_verify_key(worker):
    return worker.guest_client.credentials.verify_key


@pytest.fixture()
def guest_domain_client(root_domain_client) -> DomainClient:
    return root_domain_client.guest()


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

    state = SharedState(testrun_uid)
    KEY_CONN_STR = "mongoConnectionString"
    KEY_CLIENTS = "mongoClients"

    # start the server if it's not already running
    with state.lock:
        conn_str = state.get(KEY_CONN_STR, None)

        if not conn_str:
            conn_str = start_mongo_server(testrun_uid)
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

    # if no clients are connected, destroy the container
    if clients <= 0:
        stop_mongo_server(testrun_uid)


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
# skip compute heavy hashing during tests
SyftClientSessionCache._get_key = (
    lambda email, password, connection: f"{email}{password}{connection}"
)
