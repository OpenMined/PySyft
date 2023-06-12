# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy
from syft.util.experimental_flags import flags

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


@pytest.fixture(autouse=True)
def faker():
    return Faker()


@pytest.fixture(autouse=True)
def worker(faker):
    flags.CAN_REGISTER = False
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
    "tests.syft.messages.fixtures",
    "tests.syft.action_graph.fixtures",
]
