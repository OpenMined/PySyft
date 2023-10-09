# stdlib
import json
from pathlib import Path
from unittest import mock

# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy
from syft.protocol.data_protocol import get_data_protocol
from syft.protocol.data_protocol import stage_protocol_changes

# relative
from .syft.stores.store_fixtures_test import dict_action_store  # noqa: F401
from .syft.stores.store_fixtures_test import dict_document_store  # noqa: F401
from .syft.stores.store_fixtures_test import dict_queue_stash  # noqa: F401
from .syft.stores.store_fixtures_test import dict_store_partition  # noqa: F401
from .syft.stores.store_fixtures_test import mongo_document_store  # noqa: F401
from .syft.stores.store_fixtures_test import mongo_queue_stash  # noqa: F401
from .syft.stores.store_fixtures_test import mongo_server_mock  # noqa: F401
from .syft.stores.store_fixtures_test import mongo_store_partition  # noqa: F401
from .syft.stores.store_fixtures_test import sqlite_action_store  # noqa: F401
from .syft.stores.store_fixtures_test import sqlite_document_store  # noqa: F401
from .syft.stores.store_fixtures_test import sqlite_queue_stash  # noqa: F401
from .syft.stores.store_fixtures_test import sqlite_store_partition  # noqa: F401
from .syft.stores.store_fixtures_test import sqlite_workspace  # noqa: F401


@pytest.fixture()
def faker():
    return Faker()


def create_file(filepath: Path, data: dict):
    with open(filepath, "w") as fp:
        fp.write(json.dumps(data))


def remove_file(filepath: Path):
    filepath.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def protocol_file():
    random_name = sy.UID().to_string()
    protocol_dir = sy.SYFT_PATH / "protocol"
    file_path = protocol_dir / f"{random_name}.json"
    dp = get_data_protocol()
    create_file(filepath=file_path, data=dp.protocol_history)
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
        yield
        dp.save_history(dp.protocol_history)


@pytest.fixture(autouse=True)
def worker(faker, stage_protocol):
    return sy.Worker.named(name=faker.name())


@pytest.fixture(autouse=True)
def root_domain_client(worker):
    return worker.root_client


@pytest.fixture(autouse=True)
def root_verify_key(worker):
    return worker.root_client.credentials.verify_key


@pytest.fixture(autouse=True)
def guest_client(worker):
    return worker.guest_client


@pytest.fixture(autouse=True)
def guest_verify_key(worker):
    return worker.guest_client.credentials.verify_key


@pytest.fixture(autouse=True)
def guest_domain_client(root_domain_client):
    return root_domain_client.guest()


@pytest.fixture(autouse=True)
def document_store(worker):
    yield worker.document_store
    worker.document_store.reset()


@pytest.fixture(autouse=True)
def action_store(worker):
    return worker.action_store


__all__ = [
    "mongo_store_partition",
    "mongo_server_mock",
    "mongo_document_store",
    "mongo_queue_stash",
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
