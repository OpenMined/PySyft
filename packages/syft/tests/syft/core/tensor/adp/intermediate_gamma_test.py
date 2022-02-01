# third party
import numpy as np
import pytest

# syft absolute
from syft.core.adp.entity import DataSubjectGroup as DSG
from syft.core.adp.entity import Entity
from syft.core.adp.vm_private_scalar_manager import (
    VirtualMachinePrivateScalarManager as ScalarManager,
)
from syft.core.tensor.autodp.dp_tensor_converter import convert_to_gamma_tensor
from syft.core.tensor.autodp.intermediate_gamma import IntermediateGammaTensor as IGT
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
def dsg(ishan: Entity, traskmaster: Entity) -> DSG:
    return ishan + traskmaster


@pytest.fixture
def vsm() -> ScalarManager:
    return ScalarManager()


@pytest.fixture
def ref_square_data(dims: int) -> np.ndarray:
    return np.random.randint(low=5, high=50, size=(dims, dims), dtype=np.int32)


@pytest.fixture
def ref_data(dims: int) -> np.ndarray:
    return np.random.randint(low=5, high=50, size=(dims, dims + 1), dtype=np.int32)


@pytest.fixture
def non_square_gamma_tensor(
    ref_data: np.ndarray, ishan: Entity, traskmaster: Entity, vsm: ScalarManager
) -> IGT:
    assert ref_data.shape[0] != ref_data.shape[1]
    return SEPT(
        child=ref_data,
        min_vals=np.ones_like(ref_data),
        max_vals=np.ones_like(ref_data) * 50,
        entity=ishan,
        scalar_manager=vsm,
    ) + SEPT(
        child=ref_data,
        min_vals=np.ones_like(ref_data),
        max_vals=np.ones_like(ref_data) * 50,
        entity=traskmaster,
        scalar_manager=vsm,
    )


@pytest.fixture
def upper_bound(ref_square_data) -> np.ndarray:
    return np.ones_like(ref_square_data, dtype=np.int32) * ref_square_data.max()


@pytest.fixture
def lower_bound(ref_square_data) -> np.ndarray:
    return np.ones_like(ref_square_data, dtype=np.int32)


@pytest.fixture
def sept_ishan(
    ref_square_data,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    vsm: ScalarManager,
    ishan: Entity,
) -> SEPT:
    return SEPT(
        child=ref_square_data,
        min_vals=lower_bound,
        max_vals=upper_bound,
        entity=ishan,
        scalar_manager=vsm,
    )


@pytest.fixture
def sept_traskmaster(
    ref_square_data,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    vsm: ScalarManager,
    traskmaster: Entity,
) -> SEPT:
    return SEPT(
        child=ref_square_data,
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
    """Test that the _values() method correctly returns the np array"""
    gamma_tensor = sept_ishan + sept_traskmaster
    assert isinstance(gamma_tensor, IGT)
    target = sept_ishan.child + sept_traskmaster.child
    output = gamma_tensor._values()
    assert output.shape == gamma_tensor.shape
    assert (output == target).all()

    assert gamma_tensor == gamma_tensor_min
    output = gamma_tensor_min._values()
    assert isinstance(output, np.ndarray)
    assert output.shape == target.shape
    assert (gamma_tensor._values() == target).all()


def test_entities(sept_ishan, sept_traskmaster) -> None:
    tensor = sept_ishan + sept_traskmaster
    output = tensor._entities()
    assert isinstance(tensor, IGT)
    assert isinstance(sept_ishan.entity, Entity)
    assert isinstance(sept_traskmaster.entity, Entity)

    assert tensor.n_entities == 2
    for ent in tensor.unique_entities:
        assert isinstance(ent, Entity)
        assert ent == sept_ishan.entity or ent == sept_traskmaster.entity

    target = sept_ishan.entity + sept_traskmaster.entity
    for j in output.flatten():
        assert isinstance(j, DSG)
        assert j == target


def test_gt(
    gamma_tensor_min: IGT,
    gamma_tensor_ref: IGT,
    gamma_tensor_max: IGT,
    sept_ishan: SEPT,
    sept_traskmaster: SEPT,
    dsg: DSG,
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
    gamma_tensor_min: IGT,
    gamma_tensor_ref: IGT,
    gamma_tensor_max: IGT,
    sept_ishan: SEPT,
    sept_traskmaster: SEPT,
    dsg: DSG,
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


def test_pos(
    gamma_tensor_min: IGT,
    gamma_tensor_ref: IGT,
    gamma_tensor_max: IGT,
    dsg: DSG,
) -> None:

    assert isinstance(gamma_tensor_min, IGT)
    assert isinstance(gamma_tensor_ref, IGT)
    assert isinstance(gamma_tensor_max, IGT)

    assert (gamma_tensor_max == +gamma_tensor_max).values.all()
    assert (gamma_tensor_ref == +gamma_tensor_ref).values.all()
    assert (gamma_tensor_min == +gamma_tensor_min).values.all()

    output = +gamma_tensor_min
    for entity in output._entities().flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg


def test_neg(
    gamma_tensor_min: IGT,
    gamma_tensor_ref: IGT,
    gamma_tensor_max: IGT,
    dsg: DSG,
) -> None:

    assert isinstance(gamma_tensor_min, IGT)
    assert isinstance(gamma_tensor_ref, IGT)
    assert isinstance(gamma_tensor_max, IGT)

    assert (gamma_tensor_max * -1 == -gamma_tensor_max).values.all()
    assert (gamma_tensor_ref * -1 == -gamma_tensor_ref).values.all()
    assert (gamma_tensor_min * -1 == -gamma_tensor_min).values.all()

    output = -gamma_tensor_min
    for entity in output._entities().flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg


def test_copy(
    gamma_tensor_min: IGT,
    gamma_tensor_ref: IGT,
    gamma_tensor_max: IGT,
    dsg: DSG,
) -> None:

    output = gamma_tensor_min.copy()

    assert isinstance(gamma_tensor_min, IGT)
    assert isinstance(output, IGT)

    assert (output == gamma_tensor_min).values.all()

    for entity in output._entities().flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg


def test_eq(
    gamma_tensor_min: IGT,
    gamma_tensor_ref: IGT,
    gamma_tensor_max: IGT,
    sept_ishan: SEPT,
    sept_traskmaster: SEPT,
    dsg: DSG,
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
    gamma_tensor_min: IGT,
    gamma_tensor_ref: IGT,
    gamma_tensor_max: IGT,
    sept_ishan: SEPT,
    sept_traskmaster: SEPT,
    dsg: DSG,
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
    output = gamma_ishan >= gamma_trask + 1
    assert not output.values.any()
    for entity in output.entities.flatten():
        assert isinstance(entity, DSG)
        assert entity == dsg


def test_le(
    gamma_tensor_min: IGT,
    gamma_tensor_ref: IGT,
    gamma_tensor_max: IGT,
    sept_ishan: SEPT,
    sept_traskmaster: SEPT,
    dsg: DSG,
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
    gamma_tensor_min: IGT,
    gamma_tensor_ref: IGT,
    gamma_tensor_max: IGT,
    sept_ishan: SEPT,
    sept_traskmaster: SEPT,
    dsg: DSG,
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


@pytest.mark.skip(reason="This doesn't actually test anything")
def test_tensor_creation(sept_ishan, sept_traskmaster) -> None:
    igt = sept_ishan + sept_traskmaster
    print(f"Term tensor {igt.term_tensor.shape}")
    print(igt.term_tensor)
    print(f"Coeff tensor {igt.coeff_tensor.shape}")
    print(igt.coeff_tensor)
    print(f"Bias tensor {igt.bias_tensor.shape}")
    print(igt.bias_tensor)

    print(igt.shape)
    print(sept_ishan.shape)
    assert False


def test_transpose(non_square_gamma_tensor: IGT) -> None:
    """Test the transpose operator default behaviour (no args)"""
    output = non_square_gamma_tensor.transpose()
    original_values = non_square_gamma_tensor._values()

    # Ensure both of these have the same shapes to begin with
    assert non_square_gamma_tensor.shape == original_values.shape

    # Ensure resultant shapes are correct
    target_values = original_values.transpose()
    print(f"original shape = {non_square_gamma_tensor.shape}")
    print(f"target shape = {target_values.shape}")
    print(f"output shape = {output.shape}")
    assert output.shape == target_values.shape

    # Test to see if _values() constructs a proper shape
    output_values = output._values()
    assert output_values.shape != original_values.shape
    assert output_values.shape == target_values.shape

    # Check that transposing twice undoes the operation
    assert output.transpose() == non_square_gamma_tensor
    assert (output.transpose()._values() == original_values).all()

    # Test to see if the values have been kept the same
    print(f"Values, {type(original_values)}")
    print(original_values)
    print(f"New Values, {type(output_values)}")
    print(output_values)
    assert (output_values == target_values).all()

    old_entities = non_square_gamma_tensor._entities()
    new_entities = output._entities()
    assert old_entities.shape != new_entities.shape


def test_flatten(non_square_gamma_tensor: IGT) -> None:
    """Test the flatten operator default behaviour (no args)"""
    output = non_square_gamma_tensor.flatten()
    original_values = non_square_gamma_tensor._values()

    # Ensure both of these have the same shapes to begin with
    assert non_square_gamma_tensor.shape == original_values.shape

    # Ensure resultant shapes are correct
    target_values = original_values.flatten()
    print(f"original shape = {non_square_gamma_tensor.shape}")
    print(f"target shape = {target_values.shape}")
    print(f"output shape = {output.shape}")
    assert output.shape == target_values.shape

    # Test to see if _values() constructs a proper shape
    output_values = output._values()
    assert output_values.shape != original_values.shape
    assert output_values.shape == target_values.shape

    # Test to see if the values have been kept the same
    print(f"Values, {type(original_values)}")
    print(original_values)
    print(f"New Values, {type(output_values)}")
    print(output_values)
    assert (output_values == target_values).all()

    old_entities = non_square_gamma_tensor._entities()
    new_entities = output._entities()
    assert old_entities.shape != new_entities.shape


def test_ravel(non_square_gamma_tensor: IGT) -> None:
    """Test the ravel operator default behaviour (no args)"""
    output = non_square_gamma_tensor.ravel()
    original_values = non_square_gamma_tensor._values()

    # Ensure both of these have the same shapes to begin with
    assert non_square_gamma_tensor.shape == original_values.shape

    # Ensure resultant shapes are correct
    target_values = original_values.ravel()
    print(f"original shape = {non_square_gamma_tensor.shape}")
    print(f"target shape = {target_values.shape}")
    print(f"output shape = {output.shape}")
    assert output.shape == target_values.shape

    # Test to see if _values() constructs a proper shape
    output_values = output._values()
    assert output_values.shape != original_values.shape
    assert output_values.shape == target_values.shape

    # Test to see if the values have been kept the same
    print(f"Values, {type(original_values)}")
    print(original_values)
    print(f"New Values, {type(output_values)}")
    print(output_values)
    assert (output_values == target_values).all()

    old_entities = non_square_gamma_tensor._entities()
    new_entities = output._entities()
    assert old_entities.shape != new_entities.shape


def test_cumsum(non_square_gamma_tensor: IGT) -> None:
    """Test the cumsum operator default behaviour (no args)"""
    output = non_square_gamma_tensor.cumsum()
    original_values = non_square_gamma_tensor._values()

    # Ensure both of these have the same shapes to begin with
    assert non_square_gamma_tensor.shape == original_values.shape

    # Ensure resultant shapes are correct
    target_values = original_values.cumsum()
    print(f"original shape = {non_square_gamma_tensor.shape}")
    print(f"target shape = {target_values.shape}")
    print(f"output shape = {output.shape}")
    assert output.shape == target_values.shape

    # Test to see if _values() constructs a proper shape
    output_values = output._values()
    assert output_values.shape != original_values.shape
    assert output_values.shape == target_values.shape

    # Test to see if the values have been kept the same
    print(f"Values, {type(original_values)}")
    print(original_values)
    print(f"New Values, {type(output_values)}")
    print(output_values)
    assert (output_values == target_values).all()

    old_entities = non_square_gamma_tensor._entities()
    new_entities = output._entities()
    assert old_entities.shape != new_entities.shape


def test_cumprod(non_square_gamma_tensor: IGT) -> None:
    """Test the cumprod operator default behaviour (no args)"""
    output = non_square_gamma_tensor.cumprod()
    original_values = non_square_gamma_tensor._values()

    # Ensure both of these have the same shapes to begin with
    assert non_square_gamma_tensor.shape == original_values.shape

    # Ensure resultant shapes are correct
    target_values = original_values.cumprod()
    print(f"original shape = {non_square_gamma_tensor.shape}")
    print(f"target shape = {target_values.shape}")
    print(f"output shape = {output.shape}")
    assert output.shape == target_values.shape

    # Test to see if _values() constructs a proper shape
    output_values = output._values()
    assert output_values.shape != original_values.shape
    assert output_values.shape == target_values.shape

    # Test to see if the values have been kept the same
    print(f"Values, {type(original_values)}")
    print(original_values)
    print(f"New Values, {type(output_values)}")
    print(output_values)
    assert (output_values == target_values).all()

    old_entities = non_square_gamma_tensor._entities()
    new_entities = output._entities()
    assert old_entities.shape != new_entities.shape


def test_max(non_square_gamma_tensor: IGT) -> None:
    """Test the max operator default behaviour (no args)"""
    output = non_square_gamma_tensor.max()
    original_values = non_square_gamma_tensor._values()

    # Ensure both of these have the same shapes to begin with
    assert non_square_gamma_tensor.shape == original_values.shape

    # Ensure resultant shapes are correct
    target_values = original_values.max()
    print(f"original shape = {non_square_gamma_tensor.shape}")
    print(f"target shape = {target_values.shape}")
    print(f"output shape = {output.shape}")
    assert output.shape == target_values.shape

    # Test to see if _values() constructs a proper shape
    output_values = output._values()
    assert output_values.shape != original_values.shape
    assert output_values.shape == target_values.shape

    # Test to see if the values have been kept the same
    print(f"Values, {type(original_values)}")
    print(original_values)
    print(f"New Values, {type(output_values)}")
    print(output_values)
    assert (output_values == target_values).all()

    old_entities = non_square_gamma_tensor._entities()
    new_entities = output._entities()
    assert old_entities.shape != new_entities.shape


def test_min(non_square_gamma_tensor: IGT) -> None:
    """Test the min operator default behaviour (no args)"""
    output = non_square_gamma_tensor.min()
    original_values = non_square_gamma_tensor._values()

    # Ensure both of these have the same shapes to begin with
    assert non_square_gamma_tensor.shape == original_values.shape

    # Ensure resultant shapes are correct
    target_values = original_values.min()
    print(f"original shape = {non_square_gamma_tensor.shape}")
    print(f"target shape = {target_values.shape}")
    print(f"output shape = {output.shape}")
    assert output.shape == target_values.shape

    # Test to see if _values() constructs a proper shape
    output_values = output._values()
    assert output_values.shape != original_values.shape
    assert output_values.shape == target_values.shape

    # Test to see if the values have been kept the same
    print(f"Values, {type(original_values)}")
    print(original_values)
    print(f"New Values, {type(output_values)}")
    print(output_values)
    assert (output_values == target_values).all()

    old_entities = non_square_gamma_tensor._entities()
    new_entities = output._entities()
    assert old_entities.shape != new_entities.shape


def test_mul_public(gamma_tensor_min: IGT) -> None:
    """Test public multiplication of IGTs"""
    target = gamma_tensor_min._values() * 2
    output = gamma_tensor_min * 2
    assert isinstance(output, IGT)
    assert (output._values() == target).all()
    assert (output._min_values() == gamma_tensor_min._min_values() * 2).all()
    assert (output._max_values() == gamma_tensor_min._max_values() * 2).all()
    assert (output._entities() == gamma_tensor_min._entities()).all()


@pytest.mark.skip(reason="Still not working for IGT * IGT, or IGT * SEPT :(")
def test_mul_private(gamma_tensor_min: IGT, gamma_tensor_ref: IGT) -> None:
    """Test public multiplication of IGTs"""
    assert gamma_tensor_ref.shape == gamma_tensor_min.shape
    target = gamma_tensor_min._values() * gamma_tensor_ref._values()
    output = gamma_tensor_min * gamma_tensor_ref
    assert isinstance(output, IGT)
    assert (output._values() == target).all()
    assert (
        output._min_values()
        == gamma_tensor_min._min_values() * gamma_tensor_ref._min_values()
    ).all()
    assert (
        output._max_values()
        == gamma_tensor_min._max_values() * gamma_tensor_ref._max_values()
    ).all()
    assert (output._entities() == gamma_tensor_min._entities()).all()  # No new


@pytest.mark.skip(reason="MatMul is currently not implemented correctly for IGTs.")
def test_matmul_public(gamma_tensor_min: IGT) -> None:
    """Test public matrix multiplication of IGTs"""
    other = np.ones_like(gamma_tensor_min._values())
    target = gamma_tensor_min._values() @ other
    output = gamma_tensor_min @ other
    assert isinstance(output, IGT)
    assert (output._values() == target).all()
    assert (
        output._min_values() == gamma_tensor_min._min_values().__matmul__(other)
    ).all()
    assert (
        output._max_values() == gamma_tensor_min._max_values().__matmul__(other)
    ).all()
    assert (output._entities() == gamma_tensor_min._entities().__matmul__(other)).all()


@pytest.mark.skip(reason="MatMul is currently not implemented correctly for IGTs.")
def test_matmul_private(gamma_tensor_min: IGT, sept_ishan: SEPT) -> None:
    """Test private matrix multiplication of IGTs"""
    other = sept_ishan
    target = gamma_tensor_min._values() @ other.child
    output = gamma_tensor_min @ other
    assert isinstance(output, IGT)
    assert (output._values() == target).all()
    assert (
        output._min_values()
        == gamma_tensor_min._min_values().__matmul__(other.min_vals)
    ).all()
    assert (
        output._max_values()
        == gamma_tensor_min._max_values().__matmul__(other.max_vals)
    ).all()
    assert (
        output._entities()
        == gamma_tensor_min._entities().__matmul__(
            convert_to_gamma_tensor(other)._entities()
        )
    ).all()


def test_diagonal(gamma_tensor_min: IGT) -> None:
    """Test diagonal, without any additional arguments"""
    target = gamma_tensor_min._values().diagonal()
    output = gamma_tensor_min.diagonal()
    assert isinstance(output, IGT)
    assert (output._values() == target).all()
    assert (output._min_values() == gamma_tensor_min._min_values().diagonal()).all()
    assert (output._max_values() == gamma_tensor_min._max_values().diagonal()).all()
    assert (output._entities() == gamma_tensor_min._entities().diagonal()).all()


def test_reshape(gamma_tensor_min: IGT) -> None:
    """Note: for now, resize == reshape"""
    target = gamma_tensor_min._values().flatten()
    print(target.shape, len(target))
    output = gamma_tensor_min.reshape(int(len(target)))
    assert isinstance(output, IGT)
    assert (output._values() == target).all()


def test_resize(gamma_tensor_min: IGT) -> None:
    """Note: for now, resize == reshape"""
    target = gamma_tensor_min._values().flatten()
    print(target.shape, len(target))
    output = gamma_tensor_min.resize(int(len(target)))
    assert isinstance(output, IGT)
    assert (output._values() == target).all()


def test_compress(gamma_tensor_min: IGT) -> None:
    target = gamma_tensor_min._values().compress([False, True], axis=0)
    output = gamma_tensor_min.compress([False, True], axis=0)
    assert isinstance(output, IGT)
    assert (output._values() == target).all()


@pytest.mark.skip(reason="Temporary")
def test_abs(gamma_tensor_min: IGT) -> None:
    output = abs(gamma_tensor_min)
    assert isinstance(output, IGT)
    assert output == gamma_tensor_min


def test_all(gamma_tensor_min: IGT) -> None:
    target = True
    output = gamma_tensor_min.all()
    assert output == target


def test_any(gamma_tensor_min: IGT) -> None:
    target = True
    output = gamma_tensor_min.any()
    assert output == target


def test_swapaxes(gamma_tensor_min: IGT) -> None:
    target = gamma_tensor_min._values().swapaxes(0, 1)
    output = gamma_tensor_min.swapaxes(0, 1)
    assert (target == output._values()).all()


def test_squeeze(gamma_tensor_min: IGT) -> None:
    target = gamma_tensor_min._values().squeeze()
    output = gamma_tensor_min.squeeze()
    assert (target == output._values()).all()
