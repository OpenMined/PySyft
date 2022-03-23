# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.adp.data_subject_ledger import DataSubjectLedger
from syft.core.adp.entity import Entity
from syft.core.adp.ledger_store import DictLedgerStore
from syft.core.tensor.autodp.ndim_entity_phi import NDimEntityPhiTensor as NDEPT


@pytest.fixture
def ishan() -> Entity:
    return Entity(name="φhishan")


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
        low=-highest, high=highest, size=(dims, dims), dtype=np.int32
    )
    assert dims > 1, "Tensor not large enough"
    return reference_data


@pytest.fixture
def upper_bound(reference_data: np.ndarray, highest: int) -> np.ndarray:
    """This is used to specify the max_vals for a SEPT that is either binary or randomly generated b/w 0-1"""
    max_values = np.ones_like(reference_data) * highest
    return max_values


@pytest.fixture
def lower_bound(reference_data: np.ndarray, highest: int) -> np.ndarray:
    """This is used to specify the min_vals for a SEPT that is either binary or randomly generated b/w 0-1"""
    min_values = np.ones_like(reference_data) * -highest
    return min_values


def test_gamma_serde(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test basic serde for GammaTensor"""
    tensor1 = NDEPT(
        child=reference_data,
        entities=[0, 1],
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    gamma_tensor1 = tensor1.sum()

    ser = sy.serialize(gamma_tensor1, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert de.value == gamma_tensor1.value
    assert de.data_subjects == gamma_tensor1.data_subjects
    assert de.min_val == gamma_tensor1.min_val
    assert de.max_val == gamma_tensor1.max_val
    assert de.is_linear == gamma_tensor1.is_linear
    assert de.func == gamma_tensor1.func
    assert de.id == gamma_tensor1.id
    assert (np.asarray(de.inputs) == np.asarray(gamma_tensor1.inputs)).all()
    assert de.state.keys() == gamma_tensor1.state.keys()
    for key in de.state.keys():
        assert (de.state[key].inputs == gamma_tensor1.state[key].inputs).all()


def test_gamma_publish(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test basic serde for GammaTensor"""
    tensor1 = NDEPT(
        child=reference_data,
        entities=[0, 1],
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    gamma_tensor1 = tensor1.sum()

    print("gamma_tensor1", type(gamma_tensor1))
    ledger_store = DictLedgerStore()
    print(ledger_store.kv_store)
    user_key = b"1231"
    ledger = DataSubjectLedger.get_or_create(store=ledger_store, user_key=user_key)
    results = gamma_tensor1.publish(ledger=ledger, sigma=0.1)
    print(results, results.dtype)
    print(ledger_store.kv_store)

    # results = gamma_tensor1.publish(ledger=ledger, sigma=0.1)
    # print(results, results.dtype)
    # print(ledger_store.kv_store)

    assert False
