# stdlib
import json
import os
from pathlib import Path
import shutil
from tempfile import gettempdir
from unittest import mock

# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy
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

TMP_DIR = Path(gettempdir())
MONGODB_TMP_DIR = Path(TMP_DIR, "mongodb")
SHERLOCK_TMP_DIR = Path(TMP_DIR, "sherlock")

MONGO_PORT = 37017
MONGO_CONTAINER_PREFIX = "pytest_mongo"


@pytest.fixture()
def faker():
    return Faker()


def pytest_configure(config):
    cleanup_tmp_dirs()


def pytest_sessionfinish(session, exitstatus):
    destroy_mongo_container()


def cleanup_tmp_dirs():
    cleanup_dirs = [MONGODB_TMP_DIR, SHERLOCK_TMP_DIR]
    for _dir in cleanup_dirs:
        if _dir.exists():
            shutil.rmtree(_dir, ignore_errors=True)


def patch_protocol_file(filepath: Path):
    dp = get_data_protocol()
    original_protocol = dp.read_json(dp.file_path)
    filepath.write_text(json.dumps(original_protocol))


def remove_file(filepath: Path):
    filepath.unlink(missing_ok=True)


# Pytest hook to set the number of workers for xdist
def pytest_xdist_auto_num_workers(config):
    num = config.option.numprocesses
    if num == "auto" or num == "logical":
        return os.cpu_count()
    return None


def pytest_collection_modifyitems(items):
    for item in items:
        item_fixtures = getattr(item, "fixturenames", ())

        # group tests so that they run on the same worker
        if "mongo_client" in item_fixtures:
            item.add_marker(pytest.mark.xdist_group(name="mongo"))

        elif "redis_client" in item_fixtures:
            item.add_marker(pytest.mark.xdist_group(name="redis"))

        elif "test_sqlite_" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group(name="sqlite"))


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
def worker(faker) -> Worker:
    return sy.Worker.named(name=faker.name())


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
def redis_client_global():
    # third party
    import fakeredis

    return fakeredis.FakeRedis()


@pytest.fixture(scope="function")
def redis_client(redis_client_global, monkeypatch):
    # Current Lock implementation creates it's own StrictRedis client
    # this is a way to override all the instances of StrictRedis
    monkeypatch.setattr("redis.Redis", lambda *args, **kwargs: redis_client_global)
    monkeypatch.setattr(
        "redis.StrictRedis", lambda *args, **kwargs: redis_client_global
    )

    return redis_client_global


def start_mongo_server(port=MONGO_PORT, dbname="syft"):
    # third party
    import docker

    client = docker.from_env()
    container_name = f"{MONGO_CONTAINER_PREFIX}_{port}"

    try:
        client.containers.get(container_name)
    except docker.errors.NotFound:
        client.containers.run(
            name=container_name,
            image="mongo:7",
            ports={"27017/tcp": port},
            detach=True,
            remove=True,
            auto_remove=True,
            labels={"name": "pytest-syft"},
        )
    except Exception as e:
        raise RuntimeError(f"Docker error: {e}")

    return f"mongodb://127.0.0.1:{port}/{dbname}"


def destroy_mongo_container(port=MONGO_PORT):
    # third party
    import docker

    client = docker.from_env()
    container_name = f"{MONGO_CONTAINER_PREFIX}_{port}"

    try:
        container = client.containers.get(container_name)
        container.stop()
        container.remove()
    except docker.errors.NotFound:
        pass
    except Exception:
        pass


def get_mongo_client():
    """A race-free way to start a local mongodb server and connect to it."""

    # third party
    from filelock import FileLock
    from pymongo import MongoClient

    # file based communication for pytest-xdist workers
    lock = FileLock(str(MONGODB_TMP_DIR / "server.lock"))
    ready = Path(MONGODB_TMP_DIR / "server.ready")
    connection_string = None

    with lock:
        if ready.exists():
            # if server is ready, read the connection string from the file
            connection_string = ready.read_text()
        else:
            # start the server and write the connection string to the file
            connection_string = start_mongo_server()
            ready.write_text(connection_string)

    # connect to the local mongodb server
    client = MongoClient(connection_string)
    return client


@pytest.fixture(scope="session")
def mongo_client():
    return get_mongo_client()


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
