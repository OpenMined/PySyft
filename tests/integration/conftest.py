"""Configuration file to share fixtures across benchmarks."""

# stdlib
from typing import Any
from typing import Callable
from typing import List

# third party
import _pytest
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.node.common.client import Client

clients = []  # clients for smpc test
PORT = 9082  # domain port start


def login_clients() -> None:
    PARTIES = 3
    for i in range(PARTIES):
        try:
            client = sy.old_login(
                email="info@openmined.org",
                password="changethis",
                port=(PORT + i),
                verbose=False,
            )
            clients.append(client)
        except Exception as e:
            print(f"Cant connect to client {i}. We might have less running. {e}")


@pytest.fixture
def get_clients() -> Callable[[int], List[Any]]:
    if not clients:
        login_clients()

    def _helper_get_clients(nr_clients: int) -> List[Any]:
        return clients[:nr_clients]

    return _helper_get_clients


@pytest.fixture
def data_shape() -> np.ndarray:
    return np.random.randint(
        low=7, high=10, size=2
    )  # Somewhere between 49-100 values in a 2D array for matmul


@pytest.fixture
def data_max() -> int:
    return 10


@pytest.fixture
def reference_data(data_shape: np.ndarray, data_max: int) -> np.ndarray:
    return np.random.random(size=data_shape) * data_max


@pytest.fixture
def domain_owner() -> Client:
    return sy.old_login(email="info@openmined.org", password="changethis", port=PORT)


@pytest.fixture
def cleanup_storage(domain_owner):
    # Delete any dataset remaining before the test
    for dataset in domain_owner.datasets.all():
        domain_owner.datasets.delete(dataset_id=dataset.get("id"), skip_checks=True)

    # Execute tests
    yield

    # Delete any dataset remaining after the test
    for dataset in domain_owner.datasets.all():
        domain_owner.datasets.delete(dataset_id=dataset.get("id"), skip_checks=True)


e2e_clients = []  # clients for e2e test


def load_dataset() -> None:
    PARTIES = 2

    data = np.array([[1.2, 2.7], [3.4, 4.8]])
    # data_subjects = np.broadcast_to(np.array(DataSubjectArray(["Mars"])), data.shape)
    data_subject = "Mars"
    data = sy.Tensor(data).annotate_with_dp_metadata(0, 5, data_subject)

    for i in range(PARTIES):
        try:
            client = sy.old_login(
                email="info@openmined.org",
                password="changethis",
                port=(PORT + i),
                verbose=False,
            )

            client.load_dataset(
                assets={"data": data},
                name="Mars Data",
                description=f"{client.name}  collected Data",
            )
            assert len(client.datasets) > 0
            e2e_clients.append(client)
        except Exception as e:
            print(f"Cant connect to client {i}. We might have less running. {e}")


@pytest.fixture()
def create_data_scientist() -> Callable[[int], List[Any]]:
    if not e2e_clients:
        load_dataset()

    def _helper_create_ds(port: int, **kwargs) -> None:
        idx = port - PORT
        client = e2e_clients[idx]
        client.users.create(**kwargs)
        assert len(client.users.pandas()) > 1

    return _helper_create_ds


def pytest_configure(config: _pytest.config.Config) -> None:
    config.addinivalue_line("markers", "general: general integration tests")
    config.addinivalue_line("markers", "frontend: frontend integration tests")
    config.addinivalue_line("markers", "network: network integration tests")
    config.addinivalue_line("markers", "k8s: kubernetes integration tests")
    config.addinivalue_line("markers", "e2e: end-to-end integration tests")
    config.addinivalue_line("markers", "security: security integration tests")
    config.addinivalue_line("markers", "smpc_np: smpc integration tests")
    config.addinivalue_line("markers", "smpc_mpc_tensor: smpc integration tests")
    config.addinivalue_line("markers", "smpc_abstract: smpc integration tests")
    config.addinivalue_line("markers", "smpc_share_tensor: smpc integration tests")
    config.addinivalue_line("markers", "redis: redis tests")
    config.addinivalue_line("markers", "tff: PySyTFF integration tests")
