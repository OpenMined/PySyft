# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy


@pytest.fixture
def faker():
    return Faker()


@pytest.fixture
def worker(faker):
    return sy.Worker.named(name=faker.name())


@pytest.fixture
def document_store(worker):
    return worker.document_store


@pytest.fixture
def action_store(worker):
    return worker.action_store


pytest_plugins = [
    "tests.syft.users.fixtures",
]
