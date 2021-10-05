# third party
import numpy as np
import pytest

# syft absolute
from syft.core.adp.entity import Entity
from syft.core.adp.entity import DataSubjectGroup as DSG
from syft.core.adp.vm_private_scalar_manager import (
    VirtualMachinePrivateScalarManager as ScalarManager,
)
from syft.core.tensor.autodp.intermediate_gamma import IntermediateGammaTensor as IGT
from syft.core.tensor.autodp.row_entity_phi import RowEntityPhiTensor as REPT
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor as SEPT
from syft.core.tensor.autodp.dp_tensor_converter import convert_to_gamma_tensor


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
def dsg(ishan: Entity, traskmaster: Entity) -> DSG:
    return ishan + traskmaster


@pytest.fixture
def vsm() -> ScalarManager:
    return ScalarManager()


@pytest.fixture
def ref_data(dims: int) -> np.ndarray:
    return np.random.randint(low=5, high=50, size=(dims, dims), dtype=np.int32)


@pytest.fixture
def upper_bound(ref_data: np.ndarray) -> np.ndarray:
    return np.ones_like(ref_data, dtype=np.int32) * ref_data.max()


@pytest.fixture
def lower_bound(ref_data: np.ndarray) -> np.ndarray:
    return np.ones_like(ref_data, dtype=np.int32)


@pytest.fixture
def sept_ishan(
    ref_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    vsm: ScalarManager,
    ishan: Entity,
) -> SEPT:
    return SEPT(
        child=ref_data,
        min_vals=lower_bound,
        max_vals=upper_bound,
        entity=ishan,
        scalar_manager=vsm,
    )


@pytest.fixture
def sept_traskmaster(
    ref_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    vsm: ScalarManager,
    traskmaster: Entity,
) -> SEPT:
    return SEPT(
        child=ref_data,
        min_vals=lower_bound,
        max_vals=upper_bound,
        entity=traskmaster,
        scalar_manager=vsm,
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


def test_values(sept_ishan, sept_traskmaster, gamma_tensor_min) -> None:
    """ Test that the _values() method correctly returns the np array"""
    gamma_tensor = sept_ishan + sept_traskmaster
    assert isinstance(gamma_tensor, IGT)
    target = sept_ishan.child + sept_traskmaster.child
    output = gamma_tensor._values()
    assert output.shape == gamma_tensor.shape
    assert (output == target).all()

    assert gamma_tensor == gamma_tensor_min
    output = gamma_tensor_min._values()
    assert isinstance(output, np.ndarray)
    assert output.shape == gamma_tensor_min.shape
    assert (gamma_tensor_min._values() == target).all()


def test_entities(sept_ishan, sept_traskmaster) -> None:
    tensor = sept_ishan + sept_traskmaster
    output = tensor._entities()

    target = sept_ishan.entity + sept_traskmaster.entity
    for j in output.flatten():
        assert isinstance(j, DSG)
        assert j == target


def test_gt(
    gamma_tensor_min: IGT, gamma_tensor_ref: IGT, gamma_tensor_max: IGT, sept_ishan: SEPT, sept_traskmaster: SEPT, dsg: DSG
) -> None:

    # Private - Public
    assert isinstance(gamma_tensor_min, IGT)
    assert isinstance(gamma_tensor_ref, IGT)
    assert isinstance(gamma_tensor_max, IGT)

    assert (gamma_tensor_max > gamma_tensor_min).values.all()
    assert (gamma_tensor_ref > gamma_tensor_min).values.all()

    # Uncomment this line to ensure that the check is working properly, as
    # assert (gamma_tensor_min > gamma_tensor_ref).values.all()

    # Private - Private
    gamma_ishan = convert_to_gamma_tensor(sept_ishan)
    gamma_trask = convert_to_gamma_tensor(sept_traskmaster)

    output = gamma_ishan + 1 > gamma_trask
    assert output.values.all()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg
    output = gamma_ishan > gamma_trask
    assert not output.values.any()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg


def test_lt(
    gamma_tensor_min: IGT, gamma_tensor_ref: IGT, gamma_tensor_max: IGT, sept_ishan: SEPT, sept_traskmaster: SEPT, dsg: DSG
) -> None:

    # Private - Public
    assert isinstance(gamma_tensor_min, IGT)
    assert isinstance(gamma_tensor_ref, IGT)
    assert isinstance(gamma_tensor_max, IGT)

    assert (gamma_tensor_min < gamma_tensor_max).values.all()
    assert (gamma_tensor_min < gamma_tensor_ref).values.all()

    # Uncomment this line to ensure that the check is working properly, as
    # assert (gamma_tensor_ref < gamma_tensor_min).values.all()

    # Private - Private
    gamma_ishan = convert_to_gamma_tensor(sept_ishan)
    gamma_trask = convert_to_gamma_tensor(sept_traskmaster)

    output = gamma_ishan < gamma_trask + 1
    assert output.values.all()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg
    output = gamma_ishan < gamma_trask
    assert not output.values.any()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg


def test_eq(
    gamma_tensor_min: IGT, gamma_tensor_ref: IGT, gamma_tensor_max: IGT, sept_ishan: SEPT, sept_traskmaster: SEPT, dsg: DSG
) -> None:

    # Private - Public
    assert isinstance(gamma_tensor_min, IGT)
    assert isinstance(gamma_tensor_ref, IGT)
    assert isinstance(gamma_tensor_max, IGT)

    assert (gamma_tensor_max == gamma_tensor_max).values.all()
    assert (gamma_tensor_ref == gamma_tensor_ref).values.all()
    assert (gamma_tensor_min == gamma_tensor_min).values.all()

    # Uncomment this line to ensure that the check is working properly, as
    # assert (gamma_tensor_min == gamma_tensor_ref).values.any()

    # Private - Private
    gamma_ishan = convert_to_gamma_tensor(sept_ishan)
    gamma_trask = convert_to_gamma_tensor(sept_traskmaster)

    output = gamma_ishan == gamma_trask
    assert output.values.all()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg
    output = gamma_ishan == gamma_trask + 1
    assert not output.values.any()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg


def test_ge(
    gamma_tensor_min: IGT, gamma_tensor_ref: IGT, gamma_tensor_max: IGT, sept_ishan: SEPT, sept_traskmaster: SEPT, dsg: DSG
) -> None:

    # Private - Public
    assert isinstance(gamma_tensor_min, IGT)
    assert isinstance(gamma_tensor_ref, IGT)
    assert isinstance(gamma_tensor_max, IGT)

    assert (gamma_tensor_max >= gamma_tensor_max).values.all()
    assert (gamma_tensor_max >= gamma_tensor_ref).values.all()
    assert (gamma_tensor_min >= gamma_tensor_min).values.all()

    # Uncomment this line to ensure that the check is working properly, as
    # assert (gamma_tensor_min == gamma_tensor_ref).values.any()

    # Private - Private
    gamma_ishan = convert_to_gamma_tensor(sept_ishan)
    gamma_trask = convert_to_gamma_tensor(sept_traskmaster)

    output = gamma_ishan >= gamma_trask
    assert output.values.all()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg
    output = gamma_ishan  >= gamma_trask + 1
    assert not output.values.any()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg


def test_le(
    gamma_tensor_min: IGT, gamma_tensor_ref: IGT, gamma_tensor_max: IGT, sept_ishan: SEPT, sept_traskmaster: SEPT, dsg: DSG
) -> None:

    # Private - Public
    assert isinstance(gamma_tensor_min, IGT)
    assert isinstance(gamma_tensor_ref, IGT)
    assert isinstance(gamma_tensor_max, IGT)

    assert (gamma_tensor_max <= gamma_tensor_max).values.all()
    assert (gamma_tensor_min <= gamma_tensor_ref).values.all()
    assert (gamma_tensor_min <= gamma_tensor_max).values.all()

    # Uncomment this line to ensure that the check is working properly, as
    # assert (gamma_tensor_min == gamma_tensor_ref).values.any()

    # Private - Private
    gamma_ishan = convert_to_gamma_tensor(sept_ishan)
    gamma_trask = convert_to_gamma_tensor(sept_traskmaster)

    output = gamma_ishan <= gamma_trask
    assert output.values.all()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg
    output = gamma_ishan + 1 <= gamma_trask
    assert not output.values.any()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg


def test_ne(
    gamma_tensor_min: IGT, gamma_tensor_ref: IGT, gamma_tensor_max: IGT, sept_ishan: SEPT, sept_traskmaster: SEPT, dsg: DSG
) -> None:

    # Private - Public
    assert isinstance(gamma_tensor_min, IGT)
    assert isinstance(gamma_tensor_ref, IGT)
    assert isinstance(gamma_tensor_max, IGT)

    assert (gamma_tensor_max != gamma_tensor_min).values.all()
    assert (gamma_tensor_min != gamma_tensor_ref).values.all()
    assert not (gamma_tensor_max != gamma_tensor_max).values.all()

    # Uncomment this line to ensure that the check is working properly, as
    # assert (gamma_tensor_min == gamma_tensor_ref).values.any()

    # Private - Private
    gamma_ishan = convert_to_gamma_tensor(sept_ishan)
    gamma_trask = convert_to_gamma_tensor(sept_traskmaster)

    output = gamma_ishan + 5 != gamma_trask
    assert output.values.all()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg
    output = gamma_ishan != gamma_trask
    assert not output.values.any()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg

