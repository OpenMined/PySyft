# third party
import numpy as np
import pytest

# syft absolute
from syft.core.adp.data_subject_ledger import DataSubjectLedger
from syft.core.adp.data_subject_ledger import DictLedgerStore
from syft.core.adp.entity import Entity
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
    """Test basic serde for NDEPT"""
    tensor1 = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    gamma_tensor1 = tensor1.gamma
    print("gamma_tensor1", type(gamma_tensor1))
    ledger_store = DictLedgerStore()
    print(ledger_store.kv_store)
    user_key = b"1231"
    ledger = DataSubjectLedger.get_or_create(store=ledger_store, user_key=user_key)
    results = gamma_tensor1.publish(ledger=ledger, sigma=0.1)
    print(results)
    print(ledger_store.kv_store)

    results = gamma_tensor1.publish(ledger=ledger, sigma=0.1)

    # gamma.publish, self, sigma: Optional[float] = None, output_func: Callable = np.sum

    assert False

    # ser = sy.serialize(tensor1)
    # de = sy.deserialize(ser)

    # assert de == tensor1
    # assert (de.child == tensor1.child).all()
    # assert (de.min_vals == tensor1.min_vals).all()
    # assert (de.max_vals == tensor1.max_vals).all()
    # assert de.entities == tensor1.entities

    # assert np.shares_memory(tensor1.child, tensor1.child)
    # assert not np.shares_memory(de.child, tensor1.child)
