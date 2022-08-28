# stdlib
from typing import Any
from typing import Dict
import uuid

# third party
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


#
# @pytest.fixture
# def matmul_data(data_shape: np.ndarray, data_max: int) -> np.ndarray:
#     return np.random.random(size=(data_shape[-1], 5)) * data_max


def data_scientist(email: str, pwd: str) -> Dict["str", Any]:
    return {
        "name": "Doctor Strange",
        "email": email,
        "password": pwd,
        "budget": 9_999_999,
    }


# NOTE: We assign a high budget to the Data Scientist, as  there ShareTensor Values
# in np.int64 drastically increases the RDP Parameters, which causes a high
# privacy budget requirement.


@pytest.mark.e2e
def test_mean(
    create_data_scientist,
    domain1_port: int,
    email: str,
    password: str,
) -> None:
    """This tests DP mean, end to end"""

    create_data_scientist(domain1_port, **data_scientist(email, password))

    # Data Scientist logs in to both domains
    domain = sy.login(email=email, password=password, port=domain1_port)

    # Check that datasets are visible
    assert len(domain.datasets) > 0

    # Check PB is available
    assert domain.privacy_budget > 100
    prev_domain_budget = domain.privacy_budget

    domain_data = domain.datasets[-1]["data"]

    result = domain_data.mean()
    result.block_with_timeout(90)
    published_result1 = result.publish(sigma=2)
    published_result1.block_with_timeout(90)
    value = published_result1.get()

    assert value.shape == ()
    assert domain.privacy_budget < prev_domain_budget
    print("Published Results ", value)
