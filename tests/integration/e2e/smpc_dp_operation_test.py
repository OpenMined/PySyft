# stdlib
from typing import Any
from typing import Dict
import uuid

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy

sy.logger.remove()


@pytest.fixture
def email() -> str:
    """Unique email to run tests isolated"""
    return f"{uuid.uuid4()}@avengers.inc"


@pytest.fixture
def password() -> str:
    return "IveComeToBargain"


@pytest.fixture
def domain1_port() -> int:
    """If we change ports in the future, change here"""
    return 9082


@pytest.fixture
def domain2_port() -> int:
    """If we change ports in the future, change here"""
    return 9083


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


#
# @pytest.fixture
# def matmul_data(data_shape: np.ndarray, data_max: int) -> np.ndarray:
#     return np.random.random(size=(data_shape[-1], 5)) * data_max


def data_scientist(email: str, pwd: str) -> Dict["str", Any]:
    return {
        "name": "Doctor Strange",
        "email": email,
        "password": pwd,
        "budget": 200,
    }


def startup(
    node1_port: int,
    node2_port: int,
    data: np.ndarray,
    max_values: int,
    ds_email: str,
    ds_pwd: str,
    matmul: bool = False,
):
    """Log into both domain nodes as admin/data owner, and upload the data"""

    # Login to domain nodes
    domain1 = sy.login(
        email="info@openmined.org", password="changethis", port=node1_port
    )
    domain2 = sy.login(
        email="info@openmined.org", password="changethis", port=node2_port
    )

    # Annotate metadata
    domain1_data = sy.Tensor(data).private(
        0, max_values, entities=["Mars"] * data.shape[0], ndept=True
    )
    if matmul is False:
        domain2_data = sy.Tensor(data).private(
            0, max_values, entities=["Mars"] * data.shape[0], ndept=True
        )
    else:
        domain2_data = sy.Tensor(data.T).private(
            0, max_values, entities=["Mars"] * data.shape[0], ndept=True
        )

    # Upload data
    domain1.load_dataset(
        assets={"data": domain1_data},
        name="Mars Data",
        description="Domain 1  collected Data",
    )
    domain2.load_dataset(
        assets={"data": domain2_data},
        name="Mars Data",
        description="Domain 2 collected Data",
    )

    # Ensure datasets were loaded properly
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0
    # TODO: If these tests are run async, this might give false positives

    # Create data scientist accounts
    domain1.users.create(**data_scientist(email=ds_email, pwd=ds_pwd))
    domain2.users.create(**data_scientist(email=ds_email, pwd=ds_pwd))

    # Ensure data scientist accounts were created properly
    assert len(domain1.users.pandas()) > 1
    assert len(domain2.users.pandas()) > 1

    return None


@pytest.mark.e2e
def test_addition(
    email: str,
    domain1_port: int,
    domain2_port: int,
    reference_data: np.ndarray,
    data_max: int,
    password: str,
) -> None:
    """This tests DP and SMPC addition, end to end"""

    # Data Owner creates data, annotates it, uploads it, creates data scientist accounts
    startup(
        node1_port=domain1_port,
        node2_port=domain2_port,
        data=reference_data,
        max_values=data_max,
        ds_email=email,
        ds_pwd=password,
    )

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password, port=domain1_port)
    domain2 = sy.login(email=email, password=password, port=domain2_port)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100

    domain1_data = domain1.datasets[-1]["data"]
    domain2_data = domain2.datasets[-1]["data"]

    result = domain1_data + domain2_data
    result.block_with_timeout(60)
    published_result = result.publish(sigma=1e6)
    published_result.block_with_timeout(60)

    assert (
        published_result.shape == reference_data.shape
        or published_result.shape == reference_data.shape
    )
    assert domain1.privacy_budget < 200
    assert domain2.privacy_budget < 200
    # TODO: Figure out how to test result is reasonable
    """
    Idea: https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule#Table_of_numerical_values

    true result = reference_data * 2
    to this we add noise from Gaussian distribution with mean = 0, sigma = 100

    => ~68% of published_result values are between 2 * reference_data +/- 100
    => ~95% of published_result values are between 2 * reference_data +/- 200
    => ~99.7% of published_result values are between 2 * reference_data +/- 300
    etc etc

    We could use 5 sigma (flakiness would be expected once every 1.7M times this test is run)
    """


@pytest.mark.e2e
def test_subtraction(
    email: str,
    domain1_port: int,
    domain2_port: int,
    reference_data: np.ndarray,
    data_max: int,
    password: str,
) -> None:
    """This tests DP and SMPC subtraction, end to end"""

    # Data Owner creates data, annotates it, uploads it, creates data scientist accounts
    startup(
        node1_port=domain1_port,
        node2_port=domain2_port,
        data=reference_data,
        max_values=data_max,
        ds_email=email,
        ds_pwd=password,
    )

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password, port=domain1_port)
    domain2 = sy.login(email=email, password=password, port=domain2_port)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100

    domain1_data = domain1.datasets[-1]["data"]
    domain2_data = domain2.datasets[-1]["data"]

    result = domain1_data - domain2_data
    result.block_with_timeout(60)
    published_result = result.publish(sigma=1e6)
    published_result.block_with_timeout(60)

    assert (
        published_result.shape == reference_data.shape
        or published_result.shape == reference_data.shape
    )
    assert domain1.privacy_budget < 200
    assert domain2.privacy_budget < 200


@pytest.mark.e2e
def test_mul(
    email: str,
    domain1_port: int,
    domain2_port: int,
    reference_data: np.ndarray,
    data_max: int,
    password: str,
) -> None:
    """This tests DP and SMPC multiplication, end to end"""

    # Data Owner creates data, annotates it, uploads it, creates data scientist accounts
    startup(
        node1_port=domain1_port,
        node2_port=domain2_port,
        data=reference_data,
        max_values=data_max,
        ds_email=email,
        ds_pwd=password,
    )

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password, port=domain1_port)
    domain2 = sy.login(email=email, password=password, port=domain2_port)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100

    domain1_data = domain1.datasets[-1]["data"]
    domain2_data = domain2.datasets[-1]["data"]

    result = domain1_data * domain2_data
    result.block_with_timeout(60)
    published_result = result.publish(sigma=1e6)
    published_result.block_with_timeout(60)

    # TODO: Remove the squeeze when the vectorized_publish bug is found
    assert (
        published_result.shape == reference_data.shape
        or published_result.shape == reference_data.shape
    )
    assert domain1.privacy_budget < 200
    assert domain2.privacy_budget < 200


@pytest.mark.e2e
def test_eq(
    email: str,
    domain1_port: int,
    domain2_port: int,
    reference_data: np.ndarray,
    data_max: int,
    password: str,
) -> None:
    # Data Owner creates data, annotates it, uploads it, creates data scientist accounts
    startup(
        node1_port=domain1_port,
        node2_port=domain2_port,
        data=reference_data,
        max_values=data_max,
        ds_email=email,
        ds_pwd=password,
    )

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password, port=domain1_port)
    domain2 = sy.login(email=email, password=password, port=domain2_port)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100

    domain1_data = domain1.datasets[-1]["data"]
    domain2_data = domain2.datasets[-1]["data"]

    result = domain1_data == domain2_data
    result.block_with_timeout(60)
    published_result = result.publish(sigma=1e6)
    published_result.block_with_timeout(60)

    # TODO: Remove the squeeze when the vectorized_publish bug is found
    assert (
        published_result.shape == reference_data.shape
        or published_result.shape == reference_data.shape
    )
    assert domain1.privacy_budget < 200
    assert domain2.privacy_budget < 200


@pytest.mark.e2e
def test_ne(
    email: str,
    domain1_port: int,
    domain2_port: int,
    reference_data: np.ndarray,
    data_max: int,
    password: str,
) -> None:
    # Data Owner creates data, annotates it, uploads it, creates data scientist accounts
    startup(
        node1_port=domain1_port,
        node2_port=domain2_port,
        data=reference_data,
        max_values=data_max,
        ds_email=email,
        ds_pwd=password,
    )

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password)
    domain2 = sy.login(email=email, password=password)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100

    domain1_data = domain1[-1]["Earth Data"]
    domain2_data = domain2[-1]["Mars Data"]

    result = domain1_data != domain2_data
    result.block_with_timeout(60)
    published_result = result.publish(sigma=1e6)
    published_result.block_with_timeout(60)

    # TODO: Remove the squeeze when the vectorized_publish bug is found
    assert (
        published_result.shape == reference_data.shape
        or published_result.squeeze().shape == reference_data.shape
    )
    assert domain1.privacy_budget < 200
    assert domain2.privacy_budget < 200


@pytest.mark.e2e
def test_lt(
    email: str,
    domain1_port: int,
    domain2_port: int,
    reference_data: np.ndarray,
    data_max: int,
    password: str,
) -> None:
    # Data Owner creates data, annotates it, uploads it, creates data scientist accounts
    startup(
        node1_port=domain1_port,
        node2_port=domain2_port,
        data=reference_data,
        max_values=data_max,
        ds_email=email,
        ds_pwd=password,
    )

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password)
    domain2 = sy.login(email=email, password=password)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100

    domain1_data = domain1[-1]["Earth Data"]
    domain2_data = domain2[-1]["Mars Data"]

    result = domain1_data < domain2_data
    result.block_with_timeout(60)
    published_result = result.publish(sigma=1e6)
    published_result.block_with_timeout(60)

    # TODO: Remove the squeeze when the vectorized_publish bug is found
    assert (
        published_result.shape == reference_data.shape
        or published_result.squeeze().shape == reference_data.shape
    )
    assert domain1.privacy_budget < 200
    assert domain2.privacy_budget < 200


@pytest.mark.e2e
def test_gt(
    email: str,
    domain1_port: int,
    domain2_port: int,
    reference_data: np.ndarray,
    data_max: int,
    password: str,
) -> None:
    # Data Owner creates data, annotates it, uploads it, creates data scientist accounts
    startup(
        node1_port=domain1_port,
        node2_port=domain2_port,
        data=reference_data,
        max_values=data_max,
        ds_email=email,
        ds_pwd=password,
    )

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password)
    domain2 = sy.login(email=email, password=password)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100

    domain1_data = domain1[-1]["Earth Data"]
    domain2_data = domain2[-1]["Mars Data"]

    result = domain1_data > domain2_data
    result.block_with_timeout(60)
    published_result = result.publish(sigma=1e6)
    published_result.block_with_timeout(60)

    # TODO: Remove the squeeze when the vectorized_publish bug is found
    assert (
        published_result.shape == reference_data.shape
        or published_result.squeeze().shape == reference_data.shape
    )
    assert domain1.privacy_budget < 200
    assert domain2.privacy_budget < 200


@pytest.mark.e2e
def test_le(
    email: str,
    domain1_port: int,
    domain2_port: int,
    reference_data: np.ndarray,
    data_max: int,
    password: str,
) -> None:
    # Data Owner creates data, annotates it, uploads it, creates data scientist accounts
    startup(
        node1_port=domain1_port,
        node2_port=domain2_port,
        data=reference_data,
        max_values=data_max,
        ds_email=email,
        ds_pwd=password,
    )

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password)
    domain2 = sy.login(email=email, password=password)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100

    domain1_data = domain1[-1]["Earth Data"]
    domain2_data = domain2[-1]["Mars Data"]

    result = domain1_data <= domain2_data
    result.block_with_timeout(60)
    published_result = result.publish(sigma=1e6)
    published_result.block_with_timeout(60)

    # TODO: Remove the squeeze when the vectorized_publish bug is found
    assert (
        published_result.shape == reference_data.shape
        or published_result.squeeze().shape == reference_data.shape
    )
    assert domain1.privacy_budget < 200
    assert domain2.privacy_budget < 200


@pytest.mark.e2e
def test_ge(
    email: str,
    domain1_port: int,
    domain2_port: int,
    reference_data: np.ndarray,
    data_max: int,
    password: str,
) -> None:
    # Data Owner creates data, annotates it, uploads it, creates data scientist accounts
    startup(
        node1_port=domain1_port,
        node2_port=domain2_port,
        data=reference_data,
        max_values=data_max,
        ds_email=email,
        ds_pwd=password,
    )

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password)
    domain2 = sy.login(email=email, password=password)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100

    domain1_data = domain1[-1]["Earth Data"]
    domain2_data = domain2[-1]["Mars Data"]

    result = domain1_data >= domain2_data
    result.block_with_timeout(60)
    published_result = result.publish(sigma=1e6)
    published_result.block_with_timeout(60)

    # TODO: Remove the squeeze when the vectorized_publish bug is found
    assert (
        published_result.shape == reference_data.shape
        or published_result.squeeze().shape == reference_data.shape
    )
    assert domain1.privacy_budget < 200
    assert domain2.privacy_budget < 200


@pytest.mark.e2e
def test_matmul(
    email: str,
    domain1_port: int,
    domain2_port: int,
    reference_data: np.ndarray,
    data_max: int,
    password: str,
) -> None:
    """This tests DP and SMPC multiplication, end to end"""

    # Data Owner creates data, annotates it, uploads it, creates data scientist accounts
    startup(
        node1_port=domain1_port,
        node2_port=domain2_port,
        data=reference_data,
        max_values=data_max,
        ds_email=email,
        ds_pwd=password,
        matmul=True,
    )

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password)
    domain2 = sy.login(email=email, password=password)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100

    domain1_data = domain1[-1]["Earth Data"]
    domain2_data = domain2[-1]["Mars Data"]

    result = domain1_data @ domain2_data
    result.block_with_timeout(60)
    published_result = result.publish(sigma=1e6)
    published_result.block_with_timeout(60)

    # TODO: Remove the squeeze when the vectorized_publish bug is found
    target_shape = (reference_data.shape[0], reference_data.shape[0])
    assert (
        published_result.shape == target_shape
        or published_result.squeeze().shape == target_shape
    )
    assert domain1.privacy_budget < 200
    assert domain2.privacy_budget < 200
