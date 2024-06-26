# stdlib
from secrets import token_hex

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy


@pytest.fixture(scope="function")
def node_1():
    node = sy.orchestra.launch(
        name=token_hex(8),
        node_side_type="low",
        dev_mode=False,
        reset=True,
        local_db=True,
        create_producer=True,
        n_consumers=1,
        queue_port=None,
    )
    yield node
    node.python_node.cleanup()
    node.land()


@pytest.fixture(scope="function")
def node_2():
    node = sy.orchestra.launch(
        name=token_hex(8),
        node_side_type="high",
        dev_mode=False,
        reset=True,
        local_db=True,
        create_producer=True,
        n_consumers=1,
        queue_port=None,
    )
    yield node
    node.python_node.cleanup()
    node.land()


@pytest.fixture(scope="function")
def client_do_1(node_1):
    return node_1.login(email="info@openmined.org", password="changethis")


@pytest.fixture(scope="function")
def client_do_2(node_2):
    return node_2.login(email="info@openmined.org", password="changethis")


@pytest.fixture(scope="function")
def client_ds_1(node_1, client_do_1):
    client_do_1.register(
        name="test_user", email="test@us.er", password="1234", password_verify="1234"
    )
    return node_1.login(email="test@us.er", password="1234")


@pytest.fixture(scope="function")
def dataset_1(client_do_1):
    mock = np.array([0, 1, 2, 3, 4])
    private = np.array([5, 6, 7, 8, 9])

    dataset = sy.Dataset(
        name="my-dataset",
        description="abc",
        asset_list=[
            sy.Asset(
                name="numpy-data",
                mock=mock,
                data=private,
                shape=private.shape,
                mock_is_real=True,
            )
        ],
    )

    client_do_1.upload_dataset(dataset)
    return client_do_1.datasets[0].assets[0]


@pytest.fixture(scope="function")
def dataset_2(client_do_2):
    mock = np.array([0, 1, 2, 3, 4]) + 10
    private = np.array([5, 6, 7, 8, 9]) + 10

    dataset = sy.Dataset(
        name="my-dataset",
        description="abc",
        asset_list=[
            sy.Asset(
                name="numpy-data",
                mock=mock,
                data=private,
                shape=private.shape,
                mock_is_real=True,
            )
        ],
    )

    client_do_2.upload_dataset(dataset)
    return client_do_2.datasets[0].assets[0]
