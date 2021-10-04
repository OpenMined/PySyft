import pytest
import numpy as np
from syft.core.adp.entity import Entity
from syft.core.adp.vm_private_scalar_manager import (
    VirtualMachinePrivateScalarManager as ScalarManager,
)
from syft.core.tensor.autodp.intermediate_gamma import IntermediateGammaTensor as IGT
from syft.core.tensor.autodp.row_entity_phi import RowEntityPhiTensor as REPT
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor as SEPT


@pytest.fixture
def dims() -> int:
    return np.random.randint(low=3, high=10, dtype=np.int32)


@pytest.fixture
def ishan() -> Entity:
    return Entity(name="Ishan")


@pytest.fixture
def traskmaster() -> Entity:
    return Entity(name="Andrew")


@pytest.fixture
def vsm() -> ScalarManager:
    return ScalarManager()


@pytest.fixture
def ref_data(dims: int) -> np.ndarray:
    return np.random.randint(low=5, high=50, size=(dims, dims), dtype=np.int32)


@pytest.fixture
def upper_bound(ref_data: np.ndarray) -> np.ndarray:
    return np.ones_like(ref_data, dtype=np.int32) * max(ref_data)


@pytest.fixture
def lower_bound(ref_data: np.ndarray) -> np.ndarray:
    return np.ones_like(ref_data, dtype=np.int32)


@pytest.fixture
def sept_ishan(ref_data: np.ndarray,
               upper_bound: np.ndarray, lower_bound: np.ndarray, vsm: ScalarManager, ishan: Entity) -> SEPT:
    return SEPT(
        child=ref_data,
        min_vals=lower_bound,
        max_vals=upper_bound,
        entity=ishan,
        scalar_manager=vsm
    )


@pytest.fixture
def sept_traskmaster(ref_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray,
                     vsm: ScalarManager, traskmaster: Entity) -> SEPT:
    return SEPT(
        child=ref_data,
        min_vals=lower_bound,
        max_vals=upper_bound,
        entity=traskmaster,
        scalar_manager=vsm
    )


@pytest.fixture
def gamma_tensor_min(sept_ishan, sept_traskmaster) -> IGT:
    return sept_ishan + sept_traskmaster


@pytest.fixture
def gamma_tensor_ref(sept_ishan, sept_traskmaster) -> IGT:
    return sept_ishan + sept_traskmaster * 2


@pytest.fixture
def gamma_tensor_max(sept_ishan, sept_traskmaster) -> IGT:
    return sept_ishan + sept_traskmaster * 4


def test_gt(gamma_tensor_min: IGT, gamma_tensor_ref: IGT, gamma_tensor_max: IGT) -> None:
    assert isinstance(gamma_tensor_min, IGT)
    assert isinstance(gamma_tensor_ref, IGT)
    assert isinstance(gamma_tensor_max, IGT)


def test_lt(gamma_tensor_min: IGT, gamma_tensor_ref: IGT, gamma_tensor_max: IGT) -> None:
    pass


def test_eq(gamma_tensor_min: IGT, gamma_tensor_ref: IGT, gamma_tensor_max: IGT) -> None:
    pass


def test_ge(gamma_tensor_min: IGT, gamma_tensor_ref: IGT, gamma_tensor_max: IGT) -> None:
    pass


def test_le(gamma_tensor_min: IGT, gamma_tensor_ref: IGT, gamma_tensor_max: IGT) -> None:
    pass


def test_ne(gamma_tensor_min: IGT, gamma_tensor_ref: IGT, gamma_tensor_max: IGT) -> None:
    pass
