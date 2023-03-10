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
    return sy.Worker.named(processes=1, name=faker.name())


@pytest.fixture(autouse=True)
def document_store(worker):
    return worker.document_store


@pytest.fixture(autouse=True)
def action_store(worker):
    return worker.action_store


pytest_plugins = [
    "new_tests.users.fixtures",
]
