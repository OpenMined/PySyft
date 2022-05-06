# stdlib
from typing import Any

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.adp.data_subject_ledger import DataSubjectLedger
from syft.core.adp.ledger_store import DictLedgerStore
from syft.core.tensor.autodp.gamma_tensor import GammaTensor
from syft.core.tensor.autodp.phi_tensor import PhiTensor as PT


@pytest.fixture
def highest() -> int:
    return 50


@pytest.fixture
def lowest(highest) -> int:
    return -1 * int(highest)


@pytest.fixture
def dims() -> int:
    """This generates a random integer for the number of dimensions in our testing tensors"""
    dims = int(max(3, np.random.randint(10) + 3))  # Avoid size 0 and 1
    # Failsafe
    if dims < 2:
        dims += 3
    assert dims > 1, "Tensor not large enough for several tests."
    return dims


@pytest.fixture
def reference_data(highest, dims) -> np.ndarray:
    """This generates random data to test the equality operators"""
    reference_data = np.random.randint(
        low=-highest, high=highest, size=(dims, dims), dtype=np.int64
    )
    assert dims > 1, "Tensor not large enough"
    return reference_data


@pytest.fixture
def upper_bound(reference_data: np.ndarray, highest: int) -> np.ndarray:
    """This is used to specify the max_vals that is either binary or randomly generated b/w 0-1"""
    max_values = np.ones_like(reference_data) * highest
    return max_values


@pytest.fixture
def lower_bound(reference_data: np.ndarray, highest: int) -> np.ndarray:
    """This is used to specify the min_vals that is either binary or randomly generated b/w 0-1"""
    min_values = np.ones_like(reference_data) * -highest
    return min_values


def test_gamma_serde(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
) -> None:
    """Test basic serde for GammaTensor"""
    tensor1 = PT(
        child=reference_data,
        data_subjects=np.random.choice(["0", "1"], reference_data.shape),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    assert tensor1.data_subjects.data_subjects_indexed.shape == tensor1.child.shape
    gamma_tensor1 = tensor1.sum()

    # Checks to ensure gamma tensor was properly created
    assert isinstance(gamma_tensor1, GammaTensor)
    assert gamma_tensor1.value == tensor1.child.child.sum()

    ser = sy.serialize(gamma_tensor1, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert de.value == gamma_tensor1.value
    assert de.data_subjects == gamma_tensor1.data_subjects
    assert de.min_val == gamma_tensor1.min_val
    assert de.max_val == gamma_tensor1.max_val
    assert de.is_linear == gamma_tensor1.is_linear
    assert de.func == gamma_tensor1.func
    assert de.id == gamma_tensor1.id
    assert de.state.keys() == gamma_tensor1.state.keys()


def test_gamma_publish(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
) -> None:
    """Test basic serde for GammaTensor"""
    tensor1 = PT(
        child=reference_data,
        data_subjects=np.random.choice([0, 1], reference_data.shape),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    assert tensor1.data_subjects.data_subjects_indexed.shape == tensor1.child.shape
    gamma_tensor1 = tensor1.sum()
    assert isinstance(gamma_tensor1, GammaTensor)
    # Gamma Tensor Does not have FPT Values
    assert tensor1.child.child.sum() == gamma_tensor1.value

    ledger_store = DictLedgerStore()
    print(ledger_store.kv_store)
    user_key = b"1231"
    ledger = DataSubjectLedger.get_or_create(store=ledger_store, user_key=user_key)

    def get_budget_for_user(*args: Any, **kwargs: Any) -> float:
        return 999999

    def deduct_epsilon_for_user(*args: Any, **kwargs: Any) -> bool:
        return True

    results = gamma_tensor1.publish(
        get_budget_for_user=get_budget_for_user,
        deduct_epsilon_for_user=deduct_epsilon_for_user,
        ledger=ledger,
        sigma=0.1,
    )
    print(results)
    print(results, results.dtype)
    print(ledger_store.kv_store)
