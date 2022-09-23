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


@pytest.mark.skip(reason="Disabling due to JSON Error in github CI")
@pytest.mark.e2e
def test_addition(
    create_data_scientist,
    domain1_port: int,
    domain2_port: int,
    email: str,
    password: str,
) -> None:
    """This tests DP and SMPC addition, end to end"""

    create_data_scientist(domain1_port, **data_scientist(email, password))
    create_data_scientist(domain2_port, **data_scientist(email, password))

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password, port=domain1_port)
    domain2 = sy.login(email=email, password=password, port=domain2_port)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100
    prev_domain1_budget = domain1.privacy_budget
    prev_domain2_budget = domain2.privacy_budget

    domain1_data = domain1.datasets[-1]["data"]
    domain2_data = domain2.datasets[-1]["data"]

    result = domain1_data + domain2_data
    result.block_with_timeout(90)
    published_result = result.publish(sigma=10)
    published_result.block_with_timeout(90)

    assert published_result.shape == (2, 2)
    assert domain1.privacy_budget < prev_domain1_budget
    assert domain2.privacy_budget < prev_domain2_budget
    print("Published Result ", published_result.get())
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


@pytest.mark.skip(reason="Disabling due to JSON Error in github CI")
@pytest.mark.e2e
def test_subtraction(
    create_data_scientist,
    domain1_port: int,
    domain2_port: int,
    email: str,
    password: str,
) -> None:
    """This tests DP and SMPC subtraction, end to end"""
    create_data_scientist(domain1_port, **data_scientist(email, password))
    create_data_scientist(domain2_port, **data_scientist(email, password))

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password, port=domain1_port)
    domain2 = sy.login(email=email, password=password, port=domain2_port)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100
    prev_domain1_budget = domain1.privacy_budget
    prev_domain2_budget = domain2.privacy_budget

    domain1_data = domain1.datasets[-1]["data"]
    domain2_data = domain2.datasets[-1]["data"]

    result = domain1_data - domain2_data
    result.block_with_timeout(90)
    published_result = result.publish(sigma=10)
    published_result.block_with_timeout(90)

    assert published_result.shape == (2, 2)
    assert domain1.privacy_budget < prev_domain1_budget
    assert domain2.privacy_budget < prev_domain2_budget
    print("Published Result ", published_result.get())


@pytest.mark.skip(reason="Disabling due to JSON Error in github CI")
@pytest.mark.e2e
def test_mul(
    create_data_scientist,
    domain1_port: int,
    domain2_port: int,
    email: str,
    password: str,
) -> None:
    """This tests DP and SMPC Multiplication, end to end"""
    create_data_scientist(domain1_port, **data_scientist(email, password))
    create_data_scientist(domain2_port, **data_scientist(email, password))

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password, port=domain1_port)
    domain2 = sy.login(email=email, password=password, port=domain2_port)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100
    prev_domain1_budget = domain1.privacy_budget
    prev_domain2_budget = domain2.privacy_budget

    domain1_data = domain1.datasets[-1]["data"]
    domain2_data = domain2.datasets[-1]["data"]

    result = domain1_data * domain2_data
    result.block_with_timeout(90)
    published_result = result.publish(sigma=10)
    published_result.block_with_timeout(90)

    assert published_result.shape == (2, 2)
    assert domain1.privacy_budget < prev_domain1_budget
    assert domain2.privacy_budget < prev_domain2_budget
    print("Published Result ", published_result.get())


@pytest.mark.skip(reason="Disabling due to JSON Error in github CI")
@pytest.mark.e2e
def test_matmul(
    create_data_scientist,
    domain1_port: int,
    domain2_port: int,
    email: str,
    password: str,
) -> None:
    """This tests DP and SMPC Matrix Multiplication, end to end"""
    create_data_scientist(domain1_port, **data_scientist(email, password))
    create_data_scientist(domain2_port, **data_scientist(email, password))

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password, port=domain1_port)
    domain2 = sy.login(email=email, password=password, port=domain2_port)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100
    prev_domain1_budget = domain1.privacy_budget
    prev_domain2_budget = domain2.privacy_budget

    domain1_data = domain1.datasets[-1]["data"]
    domain2_data = domain2.datasets[-1]["data"]

    result = domain1_data @ domain2_data
    result.block_with_timeout(90)
    published_result = result.publish(sigma=10)
    published_result.block_with_timeout(90)

    assert published_result.shape == (2, 2)
    assert domain1.privacy_budget < prev_domain1_budget
    assert domain2.privacy_budget < prev_domain2_budget
    print("Published Result ", published_result.get())


@pytest.mark.skip(reason="Disabling due to JSON Error in github CI")
@pytest.mark.e2e
def test_lt(
    create_data_scientist,
    domain1_port: int,
    domain2_port: int,
    email: str,
    password: str,
) -> None:
    """This tests DP and SMPC Less than Operator, end to end"""
    create_data_scientist(domain1_port, **data_scientist(email, password))
    create_data_scientist(domain2_port, **data_scientist(email, password))

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password, port=domain1_port)
    domain2 = sy.login(email=email, password=password, port=domain2_port)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100
    prev_domain1_budget = domain1.privacy_budget
    prev_domain2_budget = domain2.privacy_budget

    domain1_data = domain1.datasets[-1]["data"]
    domain2_data = domain2.datasets[-1]["data"]

    result = domain1_data < domain2_data
    result.block_with_timeout(90)
    published_result = result.publish(sigma=10)
    published_result.block_with_timeout(90)

    assert published_result.shape == (2, 2)
    assert domain1.privacy_budget < prev_domain1_budget
    assert domain2.privacy_budget < prev_domain2_budget
    print("Published Result ", published_result.get())


@pytest.mark.skip(reason="Disabling due to JSON Error in github CI")
@pytest.mark.e2e
def test_gt(
    create_data_scientist,
    domain1_port: int,
    domain2_port: int,
    email: str,
    password: str,
) -> None:
    """This tests DP and SMPC Greater than Operator, end to end"""
    create_data_scientist(domain1_port, **data_scientist(email, password))
    create_data_scientist(domain2_port, **data_scientist(email, password))

    # Data Scientist logs in to both domains
    domain1 = sy.login(email=email, password=password, port=domain1_port)
    domain2 = sy.login(email=email, password=password, port=domain2_port)

    # Check that datasets are visible
    assert len(domain1.datasets) > 0
    assert len(domain2.datasets) > 0

    # Check PB is available
    assert domain1.privacy_budget > 100
    assert domain2.privacy_budget > 100
    prev_domain1_budget = domain1.privacy_budget
    prev_domain2_budget = domain2.privacy_budget

    domain1_data = domain1.datasets[-1]["data"]
    domain2_data = domain2.datasets[-1]["data"]

    result = domain1_data > domain2_data
    result.block_with_timeout(90)
    published_result = result.publish(sigma=10)
    published_result.block_with_timeout(90)

    assert published_result.shape == (2, 2)
    assert domain1.privacy_budget < prev_domain1_budget
    assert domain2.privacy_budget < prev_domain2_budget
    print("Published Result ", published_result.get())


# NOTE: Less than Comparison is the basic operation we use to perform other comparision
# The below comparision tests work, would be enabled after moving to
# Parallel prefix adder for comparison as they are compute intensive now.

# @pytest.mark.e2e
# def test_eq(
#     create_data_scientist,
#     domain1_port: int,
#     domain2_port: int,
#     email: str,
#     password: str,
# ) -> None:
#     """This tests DP and SMPC Equality, end to end"""
#     create_data_scientist(domain1_port, **data_scientist(email, password))
#     create_data_scientist(domain2_port, **data_scientist(email, password))

#     # Data Scientist logs in to both domains
#     domain1 = sy.login(email=email, password=password, port=domain1_port)
#     domain2 = sy.login(email=email, password=password, port=domain2_port)

#     # Check that datasets are visible
#     assert len(domain1.datasets) > 0
#     assert len(domain2.datasets) > 0

#     # Check PB is available
#     assert domain1.privacy_budget > 100
#     assert domain2.privacy_budget > 100
#     prev_domain1_budget = domain1.privacy_budget
#     prev_domain2_budget = domain2.privacy_budget

#     domain1_data = domain1.datasets[-1]["data"]
#     domain2_data = domain2.datasets[-1]["data"]

#     result = domain1_data == domain2_data
#     result.block_with_timeout(90)
#     published_result = result.publish(sigma=10)
#     published_result.block_with_timeout(20)

#     assert published_result.shape == (2, 2)
#     assert domain1.privacy_budget < prev_domain1_budget
#     assert domain2.privacy_budget < prev_domain2_budget
#     print("Published Result ", published_result.get())


# @pytest.mark.e2e
# def test_ne(
#     create_data_scientist,
#     domain1_port: int,
#     domain2_port: int,
#     email: str,
#     password: str,
# ) -> None:
#     """This tests DP and SMPC Not Equal Operator, end to end"""
#     create_data_scientist(domain1_port, **data_scientist(email, password))
#     create_data_scientist(domain2_port, **data_scientist(email, password))

#     # Data Scientist logs in to both domains
#     domain1 = sy.login(email=email, password=password, port=domain1_port)
#     domain2 = sy.login(email=email, password=password, port=domain2_port)

#     # Check that datasets are visible
#     assert len(domain1.datasets) > 0
#     assert len(domain2.datasets) > 0

#     # Check PB is available
#     assert domain1.privacy_budget > 100
#     assert domain2.privacy_budget > 100
#     prev_domain1_budget = domain1.privacy_budget
#     prev_domain2_budget = domain2.privacy_budget

#     domain1_data = domain1.datasets[-1]["data"]
#     domain2_data = domain2.datasets[-1]["data"]

#     result = domain1_data != domain2_data
#     result.block_with_timeout(90)
#     published_result = result.publish(sigma=10)
#     published_result.block_with_timeout(90)

#     assert published_result.shape == (2, 2)
#     assert domain1.privacy_budget < prev_domain1_budget
#     assert domain2.privacy_budget < prev_domain2_budget
#     print("Published Result ", published_result.get())


# @pytest.mark.e2e
# def test_le(
#     create_data_scientist,
#     domain1_port: int,
#     domain2_port: int,
#     email: str,
#     password: str,
# ) -> None:
#     """This tests DP and SMPC Less than Equal Operator, end to end"""
#     create_data_scientist(domain1_port, **data_scientist(email, password))
#     create_data_scientist(domain2_port, **data_scientist(email, password))

#     # Data Scientist logs in to both domains
#     domain1 = sy.login(email=email, password=password, port=domain1_port)
#     domain2 = sy.login(email=email, password=password, port=domain2_port)

#     # Check that datasets are visible
#     assert len(domain1.datasets) > 0
#     assert len(domain2.datasets) > 0

#     # Check PB is available
#     assert domain1.privacy_budget > 100
#     assert domain2.privacy_budget > 100
#     prev_domain1_budget = domain1.privacy_budget
#     prev_domain2_budget = domain2.privacy_budget

#     domain1_data = domain1.datasets[-1]["data"]
#     domain2_data = domain2.datasets[-1]["data"]

#     result = domain1_data <= domain2_data
#     result.block_with_timeout(90)
#     published_result = result.publish(sigma=10)
#     published_result.block_with_timeout(90)

#     assert published_result.shape == (2, 2)
#     assert domain1.privacy_budget < prev_domain1_budget
#     assert domain2.privacy_budget < prev_domain2_budget
#     print("Published Result ", published_result.get())


# @pytest.mark.e2e
# def test_ge(
#     create_data_scientist,
#     domain1_port: int,
#     domain2_port: int,
#     email: str,
#     password: str,
# ) -> None:
#     """This tests DP and SMPC Less than Equal Operator, end to end"""
#     create_data_scientist(domain1_port, **data_scientist(email, password))
#     create_data_scientist(domain2_port, **data_scientist(email, password))

#     # Data Scientist logs in to both domains
#     domain1 = sy.login(email=email, password=password, port=domain1_port)
#     domain2 = sy.login(email=email, password=password, port=domain2_port)

#     # Check that datasets are visible
#     assert len(domain1.datasets) > 0
#     assert len(domain2.datasets) > 0

#     # Check PB is available
#     assert domain1.privacy_budget > 100
#     assert domain2.privacy_budget > 100
#     prev_domain1_budget = domain1.privacy_budget
#     prev_domain2_budget = domain2.privacy_budget

#     domain1_data = domain1.datasets[-1]["data"]
#     domain2_data = domain2.datasets[-1]["data"]

#     result = domain1_data >= domain2_data
#     result.block_with_timeout(90)
#     published_result = result.publish(sigma=10)
#     published_result.block_with_timeout(90)

#     assert published_result.shape == (2, 2)
#     assert domain1.privacy_budget < prev_domain1_budget
#     assert domain2.privacy_budget < prev_domain2_budget
#     print("Published Result ", published_result.get())
