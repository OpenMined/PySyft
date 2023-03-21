# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy


@pytest.fixture(autouse=True)
def faker():
    return Faker()


@pytest.fixture(autouse=True)
def worker(faker):
    return sy.Worker.named(name=faker.name())


@pytest.fixture(autouse=True)
def root_domain_client(worker):
    return worker.root_client


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


pytest_plugins = [
    "tests.syft.users.fixtures",
]
