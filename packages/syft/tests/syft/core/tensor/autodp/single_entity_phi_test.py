# stdlib
from random import sample

# third party
import numpy as np
import pytest

# syft absolute
from syft.core.adp.entity import Entity
from syft.core.adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from syft.core.tensor.autodp.initial_gamma import IntermediateGammaTensor as IGT
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor as SEPT
from syft.core.tensor.tensor import Tensor


@pytest.fixture
def ishan() -> Entity:
    return Entity(name="Ishan")


@pytest.fixture
def traskmaster() -> Entity:
    return Entity(name="Andrew")


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


@pytest.fixture
def reference_binary_data(dims: int) -> np.ndarray:
    """Generate binary data to test the equality operators with bools"""
    binary_data = np.random.randint(2, size=(dims, dims))
    return binary_data


@pytest.fixture
def reference_scalar_manager() -> VirtualMachinePrivateScalarManager:
    """Generate a ScalarFactory that will allow GammaTensors to be created."""
    reference_scalar_manager = VirtualMachinePrivateScalarManager()
    return reference_scalar_manager


@pytest.mark.skip(
    reason="Equality works but the current method of checking it throws DeprecationWarnings"
)
def test_eq(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test equality between two identical SingleEntityPhiTensors"""
    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    # Duplicate the tensor and check if equality holds
    same_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    also_same_tensor = reference_tensor

    assert (
        reference_data == same_tensor
    ).child.all(), "Equality between identical SEPTs fails"
    assert (
        reference_tensor == also_same_tensor
    ).child.all(), "Equality between identical SEPTs fails"

    return None


def test_eq_public_shape(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test equality of SEPT tensor with Public Tensor, and with Public Tensor with a public_shape"""
    sept_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    # Without public shape
    normal_tensor: Tensor = Tensor(child=reference_data)

    # With public shape
    tensor_with_shape = Tensor(child=reference_data, public_shape=reference_data.shape)

    assert (
        sept_tensor == normal_tensor
    ).child.all(), "SEPT & Public Tensor equality failed"
    assert (
        sept_tensor == tensor_with_shape
    ).child.all(), "SEPT & Public Tensor w/ public shape equality failed"


def test_eq_diff_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    traskmaster: Entity,
) -> None:
    """Test equality between Private Tensors with different owners."""
    tensor1 = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    tensor2 = SEPT(
        child=reference_data,
        entity=traskmaster,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    result = tensor1 == tensor2
    assert isinstance(result, IGT), "Equality returns wrong value"
    assert result._values().all()
    assert (result._max_values() == np.ones_like(result._max_values())).all()
    assert (result._min_values() == np.zeros_like(result._min_values())).all()


def test_eq_ndarray(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> bool:
    """Test equality between a SEPT and a simple type (int, float, bool, np.ndarray)"""
    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    assert (
        reference_tensor == reference_data
    ).child.all(), "SEPT is apparently not equal to its underlying data."
    return True


def test_eq_bool(
    reference_binary_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> bool:
    """Test equality between a SEPT and a simple type (int, float, bool, np.ndarray)"""
    reference_tensor = SEPT(
        child=reference_binary_data,
        entity=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    assert (reference_tensor == reference_binary_data).child.all(), (
        "SEPT is apparently not equal to its underlying " "data."
    )
    return True


def test_eq_int(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> bool:
    """Test equality between a SEPT and a simple type (int, float, bool, np.ndarray)"""
    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    assert (
        reference_tensor == reference_data
    ).child.all(), "SEPT is apparently not equal to its underlying data."
    return True


def test_ne_values(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test non-equality between SEPTs with diff values but the same shape"""
    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    comparison_tensor = SEPT(
        child=reference_data + 1,
        entity=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    assert (
        reference_tensor != comparison_tensor
    ).child.any(), "SEPTs with different values are somehow equal"
    return None


@pytest.mark.skipif(dims == 1, reason="Tensor generated did not have two dimensions")
def test_ne_shapes(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    dims: int,
    highest: int,
) -> None:
    """Test non-equality between SEPTs with different shapes"""
    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    comparison_tensor = SEPT(
        child=np.random.randint(
            low=-highest, high=highest, size=(dims + 10, dims + 10), dtype=np.int32
        ),
        entity=ishan,
        max_vals=np.ones(dims + 10),
        min_vals=np.ones(dims + 10),
    )

    with pytest.raises(Exception):
        reference_tensor != comparison_tensor
    return None


def test_ne_broadcastability(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    dims: int,
) -> None:
    """Test to ensure broadcastability of array sizes works"""
    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    comparison_tensor = SEPT(
        child=np.random.random((dims, 1)),
        entity=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    assert reference_tensor != comparison_tensor, "Randomly generated tensors are equal"


def test_ne_diff_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    traskmaster: Entity,
) -> None:
    """Test non-equality between SEPTs of different entities"""
    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    comparison_tensor = SEPT(
        child=reference_data + 1,
        entity=traskmaster,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    # Ensure the raw data in both tensors is not equal
    assert (reference_tensor.child != comparison_tensor.child).all()

    result = reference_tensor != comparison_tensor
    assert isinstance(result, IGT)
    assert (
        result._values().all()
    )  # Every single value here should be 1 (True) because the values are not equal
    assert (result._max_values() == np.ones_like(result._max_values())).all()
    assert (result._min_values() == np.zeros_like(result._min_values())).all()


def test_add_wrong_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Ensure that addition with incorrect types aren't supported"""
    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    with pytest.raises(NotImplementedError):
        reference_tensor + "some string"
        reference_tensor + dict()
        # TODO: Double check how tuples behave during addition/subtraction with np.ndarrays
    return None


def test_add_simple_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    dims: int,
) -> None:
    """Test addition of a SEPT with simple types (float, ints, bools, etc)"""
    tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    random_int = np.random.randint(low=15, high=1000)
    result = tensor + random_int
    assert isinstance(result, SEPT), "SEPT + int != SEPT"
    assert (
        result.max_vals == tensor.max_vals + random_int
    ).all(), "SEPT + int: incorrect max_val"
    assert (
        result.min_vals == tensor.min_vals + random_int
    ).all(), "SEPT + int: incorrect min_val"

    random_float = random_int * np.random.rand()
    result = tensor + random_float
    assert isinstance(result, SEPT), "SEPT + float != SEPT"
    assert (
        result.max_vals == tensor.max_vals + random_float
    ).all(), "SEPT + float: incorrect max_val"
    assert (
        result.min_vals == tensor.min_vals + random_float
    ).all(), "SEPT + float: incorrect min_val"

    random_ndarray = np.random.random((dims, dims))
    result = tensor + random_ndarray
    assert isinstance(result, SEPT), "SEPT + np.ndarray != SEPT"
    # assert (result.max_vals == tensor.max_vals + random_ndarray.max()).all(), "SEPT + np.ndarray: incorrect max_val"
    # assert (result.min_vals == tensor.min_vals + random_ndarray.min()).all(), "SEPT + np.ndarray: incorrect min_val"

    return None


def test_add_tensor_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    highest: int,
    dims: int,
) -> None:
    """Test addition of a SEPT with various other kinds of Tensors"""
    # TODO: Add tests for REPT, GammaTensor, etc when those are built out.

    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    simple_tensor = Tensor(
        child=np.random.randint(
            low=-highest, high=highest, size=(dims + 10, dims + 10), dtype=np.int32
        )
    )

    with pytest.raises(NotImplementedError):
        result = reference_tensor + simple_tensor
        assert isinstance(result, SEPT), "SEPT + Tensor != SEPT"
        assert (
            result.max_vals == reference_tensor.max_vals + simple_tensor.child.max()
        ), "SEPT + Tensor: incorrect max_val"
        assert (
            result.min_vals == reference_tensor.min_vals + simple_tensor.child.min()
        ), "SEPT + Tensor: incorrect min_val"
        return None


def test_add_single_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test the addition of SEPTs"""
    tensor1 = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    tensor2 = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    result = tensor2 + tensor1
    assert isinstance(result, SEPT), "Addition of two SEPTs is wrong type"
    assert (
        result.max_vals == 2 * upper_bound
    ).all(), "Addition of two SEPTs results in incorrect max_val"
    assert (
        result.min_vals == 2 * lower_bound
    ).all(), "Addition of two SEPTs results in incorrect min_val"

    # Try with negative values
    tensor3 = SEPT(
        child=reference_data * -1.5,
        entity=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    result = tensor3 + tensor1
    assert isinstance(result, SEPT), "Addition of two SEPTs is wrong type"
    assert (
        result.max_vals == tensor3.max_vals + tensor1.max_vals
    ).all(), "SEPT + SEPT results in incorrect max_val"
    assert (
        result.min_vals == tensor3.min_vals + tensor1.min_vals
    ).all(), "SEPT + SEPT results in incorrect min_val"
    return None


@pytest.mark.skip(reason="GammaTensors have now been implemented")
def test_add_diff_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    traskmaster: Entity,
) -> None:
    """Test the addition of SEPTs"""

    tensor1 = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    tensor2 = SEPT(
        child=reference_data,
        entity=traskmaster,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    assert tensor2.entity != tensor1.entity, "Entities aren't actually different"

    with pytest.raises(NotImplementedError):
        tensor2 + tensor1
    return None


def test_add_sub_equivalence(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test that the addition of negative values is the same as subtraction."""
    tensor1 = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    tensor2 = SEPT(
        child=reference_data * -1,
        entity=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    add_result = tensor1 + tensor2
    sub_result = tensor1 - tensor1
    assert (
        add_result == sub_result
    ), "Addition of negative values does not give the same result as subtraction"
    return None


def test_add_to_gamma_tensor(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
    ishan: Entity,
    traskmaster: Entity,
) -> None:
    """Test that SEPTs with different entities create a GammaTensor when added"""
    # We have to use a reference scalar manager for now because we can't combine scalar factories yet.

    tensor1 = SEPT(
        child=reference_data,
        entity=ishan,
        max_vals=np.ones_like(reference_data),
        min_vals=np.zeros_like(reference_data),
        scalar_manager=reference_scalar_manager,
    )
    tensor2 = SEPT(
        child=reference_data,
        entity=traskmaster,
        max_vals=np.ones_like(reference_data),
        min_vals=np.zeros_like(reference_data),
        scalar_manager=reference_scalar_manager,
    )

    assert tensor2.entity != tensor1.entity, "Entities aren't actually different"
    result = tensor2 + tensor1
    assert isinstance(
        result, IGT
    ), "Addition of SEPTs with diff entities did not give GammaTensor"
    assert result.shape == tensor2.shape, "SEPT + SEPT changed shape"
    assert result.shape == tensor1.shape, "SEPT + SEPT changed shape"

    # Check that all values are as expected, and addition was conducted correctly.
    for i in range(len(result.flat_scalars)):
        assert (
            result.flat_scalars[i].value
            == tensor2.child.flatten()[i] + tensor1.child.flatten()[i]
        ), "Wrong value."
    return None


def test_sub_to_gamma_tensor(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
    ishan: Entity,
    traskmaster: Entity,
) -> None:
    """Test that SEPTs with different entities create a GammaTensor when subtracted"""
    # We have to use a reference scalar manager for now because we can't combine scalar factories yet.

    tensor1 = SEPT(
        child=reference_data,
        entity=ishan,
        max_vals=np.ones_like(reference_data),
        min_vals=np.zeros_like(reference_data),
        scalar_manager=reference_scalar_manager,
    )
    tensor2 = SEPT(
        child=reference_data,
        entity=traskmaster,
        max_vals=np.ones_like(reference_data),
        min_vals=np.zeros_like(reference_data),
        scalar_manager=reference_scalar_manager,
    )

    assert tensor2.entity != tensor1.entity, "Entities aren't actually different"
    result = tensor2 - tensor1
    assert isinstance(
        result, IGT
    ), "Addition of SEPTs with diff entities did not give GammaTensor"
    assert result.shape == tensor2.shape, "SEPT + SEPT changed shape"
    assert result.shape == tensor1.shape, "SEPT + SEPT changed shape"

    # Check that all values are as expected, and addition was conducted correctly.
    for i in range(len(result.flat_scalars)):
        assert (
            result.flat_scalars[i].value
            == tensor2.child.flatten()[i] - tensor1.child.flatten()[i]
        ), "Wrong value."
    return None


def test_pos(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
    ishan: Entity,
) -> None:
    """Ensure the __pos__ operator works as intended"""
    tensor = SEPT(
        child=reference_data,
        entity=ishan,
        max_vals=np.ones_like(reference_data),
        min_vals=np.zeros_like(reference_data),
        scalar_manager=reference_scalar_manager,
    )
    assert (
        +tensor == tensor
    ), "__pos__ failed at literally the one thing it was supposed to do."

    # Change to integer tensor
    tensor.child = tensor.child.astype("int32")
    assert +tensor == tensor, "__pos__ failed after converting floats to ints."


def test_repeat(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
    ishan: Entity,
) -> None:
    """Test that the repeat method extends a SEPT.child normally"""
    repeat_count = np.random.randint(5, 10)

    tensor = SEPT(
        child=reference_data,
        max_vals=upper_bound,
        min_vals=lower_bound,
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )
    repeated_tensor = tensor.repeat(repeat_count)  # shape = (dims*dims*repeat_count, )

    for i in range(len(tensor.child.flatten())):
        for j in range(i * repeat_count, (i + 1) * repeat_count - 1):
            assert (
                tensor.child.flatten()[i] == repeated_tensor.child[j]
            ), "Repeats did not function as intended!"


def test_repeat_axes(
    reference_data: np.ndarray,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
    ishan: Entity,
) -> None:
    """Test that the axes argument of the repeat method works as intended"""
    repeat_count = np.random.randint(5, 10)
    tensor = SEPT(
        child=reference_data,
        max_vals=np.ones_like(reference_data),
        min_vals=np.zeros_like(reference_data),
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )
    repeated_tensor = tensor.repeat(
        repeat_count, axis=1
    )  # shape = (dims*dims*repeat_count, )

    for i in range(len(tensor.child.flatten())):
        for j in range(i * repeat_count, (i + 1) * repeat_count - 1):
            assert (
                tensor.child.flatten()[i] == repeated_tensor.child.flatten()[j]
            ), "Repeats did not function as intended!"


def test_transpose_simple_types(ishan: Entity) -> None:
    """Test that if self.child can't be transposed (b/c it's an int/float/bool/etc), it isn't changed"""
    random_int = np.random.randint(low=50, high=100)
    int_tensor = SEPT(child=random_int, entity=ishan, min_vals=50, max_vals=100)
    int_tensor_transposed = int_tensor.transpose()
    assert (
        int_tensor_transposed.shape == int_tensor.shape
    ), "Transpose shape is incorrect"
    assert int_tensor_transposed.child == int_tensor.child, "Transpose: child incorrect"
    assert (
        int_tensor_transposed.min_vals == int_tensor.min_vals
    ), "Transpose: min values incorrect"
    assert (
        int_tensor_transposed.max_vals == int_tensor.max_vals
    ), "Transpose: max_values incorrect"
    # assert int_tensor_transposed.transpose() == int_tensor, "Transpose: equality error"

    random_float = random_int * np.random.random()
    float_tensor = SEPT(child=random_float, entity=ishan, min_vals=0, max_vals=100)
    float_tensor_transposed = float_tensor.transpose()
    assert (
        float_tensor_transposed.shape == float_tensor.shape
    ), "Transpose shape is incorrect"
    assert (
        float_tensor_transposed.child == float_tensor.child
    ), "Transpose: child incorrect"
    assert (
        float_tensor_transposed.min_vals == float_tensor.min_vals
    ), "Transpose: min values incorrect"
    assert (
        float_tensor_transposed.max_vals == float_tensor.max_vals
    ), "Transpose: max_values incorrect"
    # assert float_tensor_transposed == float_tensor, "Transpose: equality error"

    random_bool = np.random.choice([True, False], p=[0.5, 0.5])
    bool_tensor = SEPT(child=random_bool, entity=ishan, min_vals=0, max_vals=1)
    bool_tensor_transposed = bool_tensor.transpose()
    assert (
        bool_tensor_transposed.shape == bool_tensor.shape
    ), "Transpose shape is incorrect"
    assert (
        bool_tensor_transposed.child == bool_tensor.child
    ), "Transpose: child incorrect"
    assert (
        bool_tensor_transposed.min_vals == bool_tensor.min_vals
    ), "Transpose: min values incorrect"
    assert (
        bool_tensor_transposed.max_vals == bool_tensor.max_vals
    ), "Transpose: max_values incorrect"
    # assert bool_tensor_transposed == bool_tensor, "Transpose: equality error"
    return None


def test_transpose_square_matrix(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    dims: int,
) -> None:
    """Test transpose works on the most important use case, which is when self.child is a np.array or Tensor"""
    tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    transposed_tensor = tensor.transpose()
    assert (
        tensor.shape == transposed_tensor.shape
    ), "Transposing square matrix changed shape"
    assert (
        upper_bound.transpose() == transposed_tensor.max_vals
    ).all(), "Transpose: Incorrect max_vals"
    assert (
        lower_bound.transpose() == transposed_tensor.min_vals
    ).all(), "Transpose: Incorrect min_vals"
    assert (
        transposed_tensor.transpose() == tensor
    ), "Transposing tensor twice should return the original tensor"

    # Can't index directly into SEPT due to IndexErrors arising due to __getitem__'s effect on min_val/max_val
    for i in range(dims):
        for j in range(dims):
            assert (
                tensor.child[i, j] == transposed_tensor.child[j, i]
            ), "Transpose failed"


def test_transpose_non_square_matrix(ishan: Entity, dims: int) -> None:
    """Test transpose on SEPTs where self.child is not a square matrix"""
    rows = dims
    cols = dims + np.random.randint(low=1, high=5)
    tensor = SEPT(
        child=np.random.random((rows, cols)),
        entity=ishan,
        max_vals=np.ones(rows),
        min_vals=np.zeros(rows),
    )
    transposed_tensor = tensor.transpose()
    assert (
        tensor.shape != transposed_tensor.shape
    ), "Transposing non-square matrix did not change shape"
    assert (
        tensor.shape[::-1] == transposed_tensor.shape
    ), "Transposing non-square matrix resulted in incorrect shape"
    assert (
        np.ones((1, rows)) == transposed_tensor.max_vals
    ).all(), "Transpose: Incorrect max_vals"
    assert (
        np.zeros((1, rows)) == transposed_tensor.min_vals
    ).all(), "Transpose: Incorrect min_vals"
    assert (
        transposed_tensor.transpose() == tensor
    ), "Transposing tensor twice should return the original tensor"

    # Can't index directly into SEPT due to IndexErrors arising due to __getitem__'s effect on min_val/max_val
    for i in range(dims):
        for j in range(dims):
            assert (
                tensor.child[i, j] == transposed_tensor.child[j, i]
            ), "Transpose failed"


@pytest.mark.skip(
    reason="Test works, but checking that it works using elementwise comparison raises Deprecation Warnings"
)
def test_transpose_args(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    highest: int,
) -> None:
    """Ensure the optional arguments passed to .transpose() work as intended."""

    # Try with square matrix
    square_tensor = SEPT(
        child=reference_data, entity=ishan, min_vals=lower_bound, max_vals=upper_bound
    )
    order = list(range(len(square_tensor.shape)))
    np.random.shuffle(order)
    transposed_square_tensor = square_tensor.transpose(order)
    assert (
        square_tensor.shape == transposed_square_tensor.shape
    ), "Transposing square matrix changed shape"

    for original_index, final_index in enumerate(order):
        assert (
            square_tensor.child[:, original_index]
            == transposed_square_tensor[final_index]
        ), "Transposition failed"

    # TODO: check by reverse/undo the transpose
    # TODO: check arguments don't interfere with simple type transpose

    # Try with non-square matrix
    rows = dims
    cols = dims + np.random.randint(low=1, high=5)
    non_square_data = np.random.randint(
        low=-highest, high=highest, size=(rows, cols), dtype=np.int32
    )
    tensor = SEPT(
        child=non_square_data,
        entity=ishan,
        max_vals=np.ones_like(non_square_data) * highest,
        min_vals=np.ones_like(non_square_data) * -highest,
    )
    order = list(range(len(tensor.shape)))
    np.random.shuffle(order)
    transposed_tensor = tensor.transpose(order)
    assert (
        tensor.shape[::-1] == transposed_tensor.shape
    ), "Transposing non-square matrix resulted in incorrect shape"

    for original_index, final_index in enumerate(order):
        assert (
            tensor.child[:, original_index] == transposed_tensor[final_index]
        ), "Transposition failed"


def test_reshape(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Ensure reshape happens when it is able"""
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    new_shape = reference_data.flatten().shape[0]
    reference_tensor.reshape(new_shape)


def test_reshape_fail(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Make sure errors are raised correctly when reshape is not possible due to shape mismatch."""
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    new_shape = reference_data.flatten().shape[0]

    with pytest.raises(ValueError):
        reference_tensor.reshape(new_shape - 1)


@pytest.mark.skip(reason="Unnecessary for now, testing in reshape_fail()")
def test_reshape_simple_type() -> None:
    """Ensure reshape has no effect on simple types without shapes"""
    pass


def test_resize(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Ensure resize happens when it is able"""
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    new_shape = reference_data.flatten().shape[0]
    reference_tensor.reshape(new_shape)


def test_resize_fail(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Make sure errors are raised correctly when resize is not possible due to shape mismatch."""
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    new_shape = int(reference_data.flatten().shape[0])

    with pytest.raises(ValueError):
        reference_tensor.resize(int(new_shape - 1))
        np.resize()


def test_resize_inplace(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Ensure resize changes shape in place"""
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    initial_shape = reference_tensor.shape
    new_shape = int(reference_data.flatten().shape[0])
    assert isinstance(
        new_shape, int
    ), "new shape is not an integer, resize not possible"
    reference_tensor.resize(new_shape)
    assert (
        reference_tensor.shape != initial_shape
    ), "Resize operation failed to change shape in-place."


def test_flatten(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test that self.child can be flattened for appropriate data types"""
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    target_shape = reference_data.flatten().shape
    flattened_tensor = reference_tensor.flatten()

    assert (
        flattened_tensor.shape != reference_tensor.shape
    ), "Flattening the array really didn't do much eh"
    assert (
        flattened_tensor.shape == target_shape
    ), "Flattening did not result in the correct shape"
    assert (
        flattened_tensor == reference_data.flatten()
    ).child.all(), "Flattening changed the order of entries"


def test_ravel(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    highest: int,
) -> None:
    """Test that self.child can be ravelled for appropriate data types"""
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    target_shape = reference_data.ravel().shape
    ravelled_tensor = reference_tensor.ravel()

    assert (
        ravelled_tensor.shape != reference_tensor.shape
    ), "Ravelling the array really didn't do much eh"
    assert (
        ravelled_tensor.shape == target_shape
    ), "Ravelling did not result in the correct shape"
    assert (
        ravelled_tensor == reference_data.flatten()
    ).child.all(), "Ravelling changed the order of entries"


def test_squeeze(highest: int, ishan: Entity) -> None:
    """Test that squeeze works on an ideal case"""
    _data = np.random.randint(
        low=-highest, high=highest, size=(10, 1, 10, 1, 10), dtype=np.int32
    )
    initial_shape = _data.shape

    reference_tensor = SEPT(
        child=_data,
        max_vals=np.ones_like(_data) * highest,
        min_vals=np.ones_like(_data) * -1 * highest,
        entity=ishan,
    )

    target_data = _data.squeeze()
    target_shape = target_data.shape

    squeezed_tensor = reference_tensor.squeeze()

    assert squeezed_tensor.shape != initial_shape, "Squeezing the tensor did nothing"
    assert (
        squeezed_tensor.shape == target_shape
    ), "Squeezing the tensor gave the wrong shape"
    assert (
        squeezed_tensor == target_data
    ).child.all(), "Squeezing the tensor eliminated the wrong values"


def test_squeeze_correct_axes(highest: int, ishan: Entity) -> None:
    """Test that squeeze works on an ideal case with correct axes specified"""
    _data = np.random.randint(
        low=-1 * highest, high=highest, size=(10, 1, 10, 1, 10), dtype=np.int32
    )
    initial_shape = _data.shape

    reference_tensor = SEPT(
        child=_data,
        max_vals=np.ones_like(_data) * highest,
        min_vals=np.ones_like(_data) * -highest,
        entity=ishan,
    )

    target_data = _data.squeeze(1)
    target_shape = target_data.shape

    squeezed_tensor = reference_tensor.squeeze(1)

    assert squeezed_tensor.shape != initial_shape, "Squeezing the tensor did nothing"
    assert (
        squeezed_tensor.shape == target_shape
    ), "Squeezing the tensor gave the wrong shape"
    assert (
        squeezed_tensor == target_data
    ).child.all(), "Squeezing the tensor eliminated the wrong values"


def test_swap_axes(highest: int, ishan: Entity) -> None:
    """Test that swap_axes works on an ideal case"""
    data = np.random.randint(
        low=-highest, high=highest, size=(10, 1, 10, 1, 10), dtype=np.int32
    )
    initial_shape = data.shape

    reference_tensor = SEPT(
        child=data,
        max_vals=np.ones_like(data) * highest,
        min_vals=np.ones_like(data) * -highest,
        entity=ishan,
    )

    target_data = data.swapaxes(1, 2)
    target_shape = target_data.shape

    swapped_tensor = reference_tensor.swapaxes(1, 2)

    assert (
        swapped_tensor.shape != initial_shape
    ), "Swapping axes of  the tensor did nothing"
    assert (
        swapped_tensor.shape == target_shape
    ), "Swapping axes of  the tensor gave the wrong shape"
    assert (
        swapped_tensor == target_data
    ).child.all(), "Swapping axes of  the tensor eliminated the wrong values"


@pytest.mark.skipif(dims == 1, reason="Tensor generated did not have two dimensions")
def test_compress(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    result = reference_tensor.compress([0, 1])
    assert result == reference_data.compress(
        [0, 1]
    ), "Compress did not work as expected"

    result2 = reference_tensor.compress([0, 1], axis=1)
    assert result2 == reference_data.compress(
        [0, 1], axis=1
    ), "Compress did not work as expected"


# @pytest.mark.skipif(dims == 1, reason="Tensor generated did not have two dimensions")
@pytest.mark.skip(reason="Not supporting for 0.6.0 release")
def test_partition(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    k = 1

    reference_tensor.partition(k)
    assert reference_tensor != reference_data, "Partition did not work as expected"
    reference_data.partition(k)
    assert reference_tensor == reference_data, "Partition did not work as expected"


# @pytest.mark.skipif(dims == 1, reason="Tensor generated did not have two dimensions")
@pytest.mark.skip(reason="Not supporting for 0.6.0 release")
def test_partition_axis(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    k = 1

    reference_tensor.partition(k, axis=1)
    assert reference_tensor != reference_data, "Partition did not work as expected"
    reference_data.partition(k, axis=1)
    assert reference_tensor == reference_data, "Partition did not work as expected"


@pytest.mark.skip(reason="IGT equality bugs")
def test_mul(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
    ishan: Entity,
    traskmaster: Entity,
) -> None:
    """ """
    sept1 = SEPT(
        child=reference_data,
        min_vals=lower_bound,
        max_vals=upper_bound,
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )
    sept2 = SEPT(
        child=reference_data,
        min_vals=lower_bound,
        max_vals=upper_bound,
        entity=traskmaster,
        scalar_manager=reference_scalar_manager,
    )

    # Public-Public
    output = sept2 * sept2
    assert output.shape == sept2.shape
    # assert (output.min_vals == sept2.min_vals * sept2.min_vals).all()
    # assert (output.max_vals == sept2.max_vals * sept2.max_vals).all()
    assert (output.child == sept2.child * sept2.child).all()

    # Public - Private
    output: IGT = sept2 * sept1
    assert output.shape == sept2.shape
    # assert (output.min_vals == sept1.min_vals * sept2.min_vals).all()
    # assert (output.max_vals == sept1.max_vals * sept2.max_vals).all()
    values = np.array([i.value for i in output.flat_scalars], dtype=np.int32).reshape(
        output.shape
    )
    target = sept1.child + sept2.child
    assert target.shape == values.shape
    assert (sept1.child + sept2.child == values).all()

    # assert output.child == sept1.child * sept2.child
    return None


def test_neg(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test __neg__"""
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )
    negative_tensor = reference_tensor.__neg__()
    assert (negative_tensor.child == reference_tensor.child * -1).all()
    assert (negative_tensor.min_vals == reference_tensor.max_vals * -1).all()
    assert (negative_tensor.max_vals == reference_tensor.min_vals * -1).all()
    assert negative_tensor.shape == reference_tensor.shape


def test_and(reference_binary_data: np.ndarray, ishan: Entity) -> None:
    """Test bitwise and"""
    reference_tensor = SEPT(
        child=reference_binary_data,
        max_vals=np.ones_like(reference_binary_data),
        min_vals=np.zeros_like(reference_binary_data),
        entity=ishan,
    )
    output = reference_tensor & False
    target = reference_binary_data & False
    assert (output.child == target).all()


def test_or(reference_binary_data: np.ndarray, ishan: Entity) -> None:
    """Test bitwise or"""
    reference_tensor = SEPT(
        child=reference_binary_data,
        max_vals=np.ones_like(reference_binary_data),
        min_vals=np.zeros_like(reference_binary_data),
        entity=ishan,
    )
    output = reference_tensor | False
    target = reference_binary_data | False
    assert (output.child == target).all()


def test_cumsum(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test cumsum operator without any additional arguments"""
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )
    output = reference_tensor.cumsum()
    target = reference_data.cumsum()
    assert (output.child == target).all()
    assert (output.min_vals == lower_bound.cumsum()).all()
    assert (output.max_vals == upper_bound.cumsum()).all()
    # assert output.shape == reference_tensor.flatten().shape


def test_cumprod(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test cumprod operator without any additional arguments"""
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )
    output = reference_tensor.cumprod()
    target = reference_data.cumprod()
    assert (output.child == target).all()
    assert (output.min_vals == lower_bound.cumprod()).all()
    assert (output.max_vals == upper_bound.cumprod()).all()
    # assert output.shape == reference_tensor.flatten().shape


def test_floordiv_array(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    traskmaster: Entity,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
) -> None:
    """Test floordiv"""
    reference_tensor = SEPT(
        child=reference_data,
        max_vals=upper_bound,
        min_vals=lower_bound,
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )

    other = np.ones_like(reference_data)
    output = reference_tensor // other
    assert isinstance(output, SEPT)
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data).all()
    assert (output.max_vals == upper_bound).all()
    assert (output.min_vals == lower_bound).all()


def test_floordiv_sept(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    traskmaster: Entity,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
) -> None:
    """Test floordiv with public SEPT"""
    reference_tensor = SEPT(
        child=reference_data,
        max_vals=upper_bound,
        min_vals=lower_bound,
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )

    other = SEPT(
        child=np.ones_like(reference_data),
        max_vals=np.ones_like(reference_data) * 5,
        min_vals=np.ones_like(reference_data),
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )
    output = reference_tensor // other
    assert isinstance(output, SEPT)
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data).all()
    assert (output.max_vals == upper_bound // 5).all()
    assert (output.min_vals == lower_bound).all()  # Beware of division by 0 error


def test_min(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test min(), without any arguments"""
    tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )
    output = tensor.min()
    target = reference_data.min()
    assert output.child == target
    assert output.min_vals == lower_bound.min()
    assert output.max_vals == upper_bound.min()


def test_min_args(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    highest: int,
) -> None:
    """Test the hundred different args that exist for min()"""
    tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    # Test axis
    output = tensor.min(axis=1)
    target = reference_data.min(axis=1)
    assert (output.child == target).all()
    assert (output.min_vals == lower_bound.min(axis=1)).all()
    assert (output.max_vals == upper_bound.min(axis=1)).all()

    # Test keepdims
    output = tensor.min(keepdims=True)
    target = reference_data.min(keepdims=True)
    assert (output.child == target).all()
    assert (output.min_vals == lower_bound.min(keepdims=True)).all()
    assert (output.max_vals == upper_bound.min(keepdims=True)).all()

    # Test initial
    output = tensor.min(initial=-highest)
    target = reference_data.min(initial=-highest)
    assert (output.child == target).all()
    assert (output.min_vals == lower_bound.min(initial=-highest)).all()
    assert (output.max_vals == upper_bound.min(initial=-highest)).all()


def test_max(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test min"""
    tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )
    output = tensor.max()
    target = reference_data.max()
    assert output.child == target
    assert output.min_vals == lower_bound.max()
    assert output.max_vals == upper_bound.max()


def test_max_args(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    highest: int,
) -> None:
    """Test the hundred different args that exist for max()"""
    tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    # Test axis
    output = tensor.max(axis=1)
    target = reference_data.max(axis=1)
    assert (output.child == target).all()
    assert (output.min_vals == lower_bound.max(axis=1)).all()
    assert (output.max_vals == upper_bound.max(axis=1)).all()

    # Test keepdims
    output = tensor.max(keepdims=True)
    target = reference_data.max(keepdims=True)
    assert (output.child == target).all()
    assert (output.min_vals == lower_bound.max(keepdims=True)).all()
    assert (output.max_vals == upper_bound.max(keepdims=True)).all()

    # Test initial
    output = tensor.max(initial=-highest)
    target = reference_data.max(initial=-highest)
    assert (output.child == target).all()
    assert (output.min_vals == lower_bound.max(initial=-highest)).all()
    assert (output.max_vals == upper_bound.max(initial=-highest)).all()


@pytest.mark.skip(reason="Not supporting for 0.6.0 release")
def test_mod_array(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    traskmaster: Entity,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
) -> None:
    """Test mod"""
    reference_tensor = SEPT(
        child=reference_data,
        max_vals=upper_bound,
        min_vals=lower_bound,
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )

    other = np.ones_like(reference_data)
    output = reference_tensor % other
    assert isinstance(output, SEPT)
    assert output.shape == reference_tensor.shape
    assert (output.child == np.zeros_like(reference_data)).all()
    assert (output.max_vals == np.zeros_like(upper_bound)).all()
    assert (output.min_vals == np.zeros_like(lower_bound)).all()

    other = np.ones_like(reference_data) * 4
    output = reference_tensor % other
    assert isinstance(output, SEPT)
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data % 4).all()
    assert (output.max_vals == upper_bound % 4).all()
    assert (output.min_vals == lower_bound % 4).all()


@pytest.mark.skip(reason="Not supporting for 0.6.0 release")
def test_mod_sept(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    traskmaster: Entity,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
) -> None:
    """Test mod with public SEPT"""
    reference_tensor = SEPT(
        child=reference_data,
        max_vals=upper_bound,
        min_vals=lower_bound,
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )

    other = SEPT(
        child=np.ones_like(reference_data),
        max_vals=np.ones_like(reference_data) * 5,
        min_vals=np.ones_like(reference_data),
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )
    output = reference_tensor % other
    assert isinstance(output, SEPT)
    assert output.shape == reference_tensor.shape
    assert (output.child == np.zeros_like(reference_data)).all()
    assert (output.max_vals == np.zeros_like(upper_bound) % 5).all()
    assert (
        output.min_vals == np.zeros_like(lower_bound)
    ).all()  # Beware of division by 0 error

    other = SEPT(
        child=np.ones_like(reference_data) * 6,
        max_vals=np.ones_like(reference_data) * 6,
        min_vals=np.ones_like(reference_data),
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )
    output = reference_tensor % other
    assert isinstance(output, SEPT)
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data % 6).all()
    assert (output.max_vals == upper_bound % 6).all()
    assert (
        output.min_vals == np.zeros_like(lower_bound)
    ).all()  # Beware of division by 0 error


@pytest.mark.skip(reason="Not supporting for 0.6.0 release")
def test_divmod_array(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    traskmaster: Entity,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
) -> None:
    reference_tensor = SEPT(
        child=reference_data,
        max_vals=upper_bound,
        min_vals=lower_bound,
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )

    other = np.ones_like(reference_data) * 4
    quotient, remainder = reference_tensor.__divmod__(other)
    assert isinstance(quotient, SEPT)
    assert isinstance(remainder, SEPT)
    assert quotient.shape == reference_tensor.shape
    assert remainder.shape == reference_tensor.shape
    assert (quotient.child == reference_data // 4).all()
    assert (remainder.child == reference_data % 4).all()
    assert (quotient.max_vals == upper_bound // 4).all()
    assert (remainder.max_vals == upper_bound % 4).all()
    assert (quotient.min_vals == lower_bound // 4).all()
    assert (remainder.min_vals == lower_bound % 4).all()


@pytest.mark.skip(reason="Not supporting for 0.6.0 release")
def test_divmod_sept(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    traskmaster: Entity,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
) -> None:
    reference_tensor = SEPT(
        child=reference_data,
        max_vals=upper_bound,
        min_vals=lower_bound,
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )

    other = SEPT(
        child=np.ones_like(reference_data),
        max_vals=np.ones_like(reference_data) * 5,
        min_vals=np.ones_like(reference_data),
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )
    quotient, remainder = reference_tensor.__divmod__(other)
    assert isinstance(quotient, SEPT)
    assert isinstance(remainder, SEPT)
    assert quotient.shape == reference_tensor.shape
    assert remainder.shape == reference_tensor.shape
    assert (quotient.child == reference_data).all()
    assert (remainder.child == np.zeros_like(reference_data)).all()
    assert (quotient.max_vals == upper_bound // 5).all()
    assert (remainder.max_vals == np.zeros_like(upper_bound)).all()
    assert (quotient.min_vals == lower_bound).all()  # Beware of division by 0 error
    assert (
        remainder.min_vals == np.zeros_like(lower_bound)
    ).all()  # Beware of division by 0 error

    other = SEPT(
        child=np.ones_like(reference_data) * 6,
        max_vals=np.ones_like(reference_data) * 6,
        min_vals=np.ones_like(reference_data),
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )

    quotient, remainder = reference_tensor.__divmod__(other)
    assert isinstance(quotient, SEPT)
    assert isinstance(remainder, SEPT)
    assert quotient.shape == reference_tensor.shape
    assert remainder.shape == reference_tensor.shape
    assert (quotient.child == reference_data // 6).all()
    assert (remainder.child == reference_data % 6).all()
    assert (quotient.max_vals == upper_bound // 6).all()
    assert (remainder.max_vals == upper_bound % 6).all()
    assert (quotient.min_vals == lower_bound).all()  # Beware of division by 0 error
    assert (
        remainder.min_vals == np.zeros_like(lower_bound)
    ).all()  # Beware of division by 0 error


def test_matmul_array(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    traskmaster: Entity,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
) -> None:
    reference_tensor = SEPT(
        child=reference_data,
        max_vals=upper_bound,
        min_vals=lower_bound,
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )
    other = np.ones_like(reference_data.T) * 5
    output = reference_tensor.__matmul__(other)
    assert output.shape[0] == reference_data.shape[0]
    assert output.shape[1] == other.shape[1]
    assert (output.child == reference_data.__matmul__(other)).all()


def test_matmul_sept(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    traskmaster: Entity,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
) -> None:
    reference_tensor = SEPT(
        child=reference_data,
        max_vals=upper_bound,
        min_vals=lower_bound,
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )
    data = np.ones_like(reference_data.T) * 5
    other = SEPT(
        child=data,
        max_vals=np.ones_like(data) * 10,
        min_vals=np.ones_like(data),
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )
    output = reference_tensor.__matmul__(other)
    assert output.shape[0] == reference_data.shape[0]
    assert output.shape[1] == other.shape[1]
    assert (output.child == reference_data.__matmul__(other.child)).all()


def test_trace(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> None:
    """Test whether the trace() method works"""
    tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    output = tensor.trace()
    target = reference_data.trace()
    assert (output.child == target).all()
    assert output.min_vals == lower_bound.trace()
    assert output.max_vals == upper_bound.trace()


def test_prod(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> None:
    """Test whether the prod() method works"""
    tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    output = tensor.prod()
    target = reference_data.prod()
    assert (output.child == target).all()
    assert output.min_vals == lower_bound.prod()
    assert output.max_vals == upper_bound.prod()


def test_round(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )
    output = reference_tensor.round(decimals=0)
    target = reference_data.astype(dtype=np.int32)
    assert (output.child == target).all()
    assert (output.min_vals == lower_bound.astype(dtype=np.int32)).all()
    assert (output.max_vals == upper_bound.astype(dtype=np.int32)).all()


def test_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test that the n_entities works for SEPTs"""
    tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )
    assert isinstance(tensor, SEPT)
    assert tensor.n_entities == 1


# End of Ishan's tests


@pytest.fixture
def ent() -> Entity:
    return Entity(name="test")


@pytest.fixture
def ent2() -> Entity:
    return Entity(name="test2")


@pytest.fixture
def reference_sept(entity_name, low, high) -> SEPT:
    """This is used to generate a Single Entity Phi Tensor with random values in [low, high)"""
    child = np.random.randint(low=low, high=high, size=(dims, dims))
    max_vals = np.full((dims, dims), high - 1, dtype=np.int32)
    min_vals = np.full((dims, dims), low, dtype=np.int32)
    entity = Entity(name=entity_name)
    return SEPT(child=child, entity=entity, max_vals=max_vals, min_vals=min_vals)


def test_le_same_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
) -> None:
    tensor1 = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    # same data, same entity
    tensor2 = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )

    # different data, same entity
    tensor3 = SEPT(
        child=reference_data + 1, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )

    assert tensor1.__le__(tensor2).child.all()
    assert tensor1.__le__(tensor3).child.all()
    assert tensor1.__le__(reference_data).child.all()


def test_le_diff_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
    ent2: Entity,
) -> None:
    tensor1 = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    # same data, different entity
    tensor2 = SEPT(
        child=reference_data, entity=ent2, max_vals=upper_bound, min_vals=lower_bound
    )

    result = tensor1 <= tensor2
    assert isinstance(result, IGT)
    assert result._values().all()
    assert (result._max_values() == np.ones_like(result._max_values())).all()
    assert (result._min_values() == np.zeros_like(result._min_values())).all()


def test_ge_same_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
) -> None:
    tensor1 = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    # same data, same entity
    tensor2 = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    # different data, same entity
    tensor3 = SEPT(
        child=reference_data + 1, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )

    assert tensor1.__ge__(tensor2).child.all()
    assert tensor3.__ge__(tensor1).child.all()
    assert tensor1.__ge__(reference_data).child.all()


def test_ge_diff_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
    ent2: Entity,
) -> None:
    tensor1 = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    # same data, different entity
    tensor2 = SEPT(
        child=reference_data, entity=ent2, max_vals=upper_bound, min_vals=lower_bound
    )

    result = tensor1 <= tensor2
    assert isinstance(result, IGT)
    assert result._values().all()
    assert (result._max_values() == np.ones_like(result._max_values())).all()
    assert (result._min_values() == np.zeros_like(result._min_values())).all()


def test_lt_same_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
) -> None:
    tensor1 = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    # same data, same entity
    tensor2 = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    # different data, same entity
    tensor3 = SEPT(
        child=reference_data + 1, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )

    assert not tensor1.__lt__(tensor2).child.all()
    assert tensor1.__lt__(tensor3).child.all()
    assert not tensor1.__lt__(reference_data).child.all()


@pytest.fixture
def sept_ishan(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
    ishan: Entity,
) -> SEPT:
    return SEPT(
        child=reference_data,
        min_vals=lower_bound,
        max_vals=upper_bound,
        entity=ishan,
        scalar_manager=reference_scalar_manager,
    )


@pytest.fixture
def sept_traskmaster(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
    traskmaster: Entity,
) -> SEPT:
    return SEPT(
        child=reference_data,
        min_vals=lower_bound,
        max_vals=upper_bound,
        entity=traskmaster,
        scalar_manager=reference_scalar_manager,
    )


def test_lt_diff_entities_ishan(sept_ishan: SEPT, sept_traskmaster: SEPT) -> None:
    output = sept_ishan < sept_traskmaster + 1
    assert output.values.all()


def test_gt_diff_entities_ishan(sept_ishan: SEPT, sept_traskmaster: SEPT) -> None:
    output = sept_ishan + 1 > sept_traskmaster
    assert output.values.all()


@pytest.mark.skip(
    reason="This is broken for some reason, and has been replaced by the two tests above"
)
def test_lt_diff_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
    ent2: Entity,
) -> None:
    tensor1 = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )

    tensor2 = SEPT(
        child=reference_data + 10,
        entity=ent2,
        max_vals=upper_bound + 10,
        min_vals=lower_bound + 10,
    )

    assert (tensor1.child < tensor2.child).all()

    result = tensor1 < tensor2
    assert isinstance(result, IGT)
    assert (tensor1.gamma.values < tensor2.gamma.values).all()

    # I don't know why this happens but if you remove the next two lines, the test fails
    result2 = tensor1 == tensor2
    assert result2._values().all()

    assert result._values().all()
    assert (result._max_values() == np.ones_like(result._max_values())).all()
    assert (result._min_values() == np.zeros_like(result._min_values())).all()


def test_gt_same_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
) -> None:
    tensor1 = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    # same data, same entity
    tensor2 = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )

    # different data, same entity
    tensor3 = SEPT(
        child=reference_data + 1, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )

    assert not tensor1.__gt__(tensor2).child.all()
    assert tensor3.__gt__(tensor1).child.all()
    assert not tensor1.__gt__(reference_data).child.all()


@pytest.mark.skip(reason="Fails for some mysterious reason, replaced by a test above")
def test_gt_diff_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
    ent2: Entity,
) -> None:
    tensor1 = SEPT(
        child=reference_data + 1, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    # same data, different entity
    tensor2 = SEPT(
        child=reference_data, entity=ent2, max_vals=upper_bound, min_vals=lower_bound
    )

    result = tensor1 > tensor2
    assert isinstance(result, IGT)

    result2 = tensor1 == tensor2
    assert result2._values().all()

    assert result._values().all()
    assert (result._max_values() == np.ones_like(result._max_values())).all()
    assert (result._min_values() == np.zeros_like(result._min_values())).all()


def test_clip(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
    highest,
) -> None:
    tensor = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    clip_min = np.random.randint(-highest, highest / 2)
    clip_max = np.random.randint(highest / 2, highest)

    clipped_tensor1 = tensor.clip(clip_min, clip_max)
    clipped_tensor2 = tensor.clip(clip_max, clip_min)
    clipped_tensor3 = tensor.clip(clip_min, None)

    assert (
        (clipped_tensor1.child >= clip_min) & (clipped_tensor1.child <= clip_max)
    ).all()
    assert (clipped_tensor2.child == clip_min).all()
    assert (clipped_tensor3.child >= clip_min).all()


@pytest.fixture
def pos_reference_data(highest, dims) -> np.ndarray:
    reference_data = np.random.randint(
        low=1, high=highest, size=(dims, dims), dtype=np.int32
    )
    assert dims > 1, "Tensor not large enough"
    return reference_data


@pytest.fixture
def pos_upper_bound(pos_reference_data: np.ndarray, highest: int) -> np.ndarray:
    max_values = np.ones_like(pos_reference_data) * highest
    return max_values


@pytest.fixture
def pos_lower_bound(pos_reference_data: np.ndarray) -> np.ndarray:
    min_values = np.ones_like(pos_reference_data) * 1
    return min_values


def test_any(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    pos_reference_data: np.ndarray,
    pos_upper_bound: np.ndarray,
    pos_lower_bound: np.ndarray,
    ent: Entity,
) -> None:
    zeros_tensor = SEPT(
        child=reference_data * 0, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    pos_tensor = SEPT(
        child=pos_reference_data,
        entity=ent,
        max_vals=pos_upper_bound,
        min_vals=pos_lower_bound,
    )
    assert not zeros_tensor.any().child
    assert pos_tensor.any().child


def test_all(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    pos_reference_data: np.ndarray,
    pos_upper_bound: np.ndarray,
    pos_lower_bound: np.ndarray,
    ent: Entity,
) -> None:
    zeros_tensor = SEPT(
        child=reference_data * 0, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    pos_tensor = SEPT(
        child=pos_reference_data,
        entity=ent,
        max_vals=pos_upper_bound,
        min_vals=pos_lower_bound,
    )
    assert not zeros_tensor.all().child
    assert pos_tensor.all().child


def test_abs(
    pos_reference_data: np.ndarray,
    pos_upper_bound: np.ndarray,
    pos_lower_bound: np.ndarray,
    ent: Entity,
) -> None:
    tensor = SEPT(
        child=pos_reference_data,
        entity=ent,
        max_vals=pos_upper_bound,
        min_vals=pos_lower_bound,
    )
    neg_tensor = SEPT(
        child=pos_reference_data * -1,
        entity=ent,
        max_vals=pos_upper_bound,
        min_vals=pos_lower_bound,
    )
    assert (neg_tensor.abs().child == tensor.child).all()


def test_pow(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
) -> None:
    rand_pow = np.random.randint(1, 10)
    tensor = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    pow_tensor = SEPT(
        child=reference_data**rand_pow,
        entity=ent,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    assert (tensor.pow(rand_pow).child == pow_tensor.child).all()


def test_sum(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
    dims: int,
) -> None:
    zeros_tensor = SEPT(
        child=reference_data * 0, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    tensor = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    tensor_sum = 0
    for i in range(dims):
        for j in range(dims):
            tensor_sum += tensor.child[i, j]

    assert tensor.sum().child == tensor_sum
    assert zeros_tensor.sum().child == 0


def test_copy(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
) -> None:
    tensor = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    tensor_copy = tensor.copy()
    assert (
        (tensor_copy.child == tensor.child).all()
        & (tensor_copy.min_vals == tensor.min_vals).all()
        & (tensor_copy.max_vals == tensor.max_vals).all()
    )


def test_take(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
    dims: int,
) -> None:
    tensor = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    indices = sample(range(dims), dims)
    tensor_take = tensor.take(indices)
    assert (tensor_take.child == tensor.child[0][indices]).all()


def test_diagonal(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ent: Entity,
    dims: int,
) -> None:
    tensor = SEPT(
        child=reference_data, entity=ent, max_vals=upper_bound, min_vals=lower_bound
    )
    tensor_diagonal = tensor.diagonal()
    for i in range(dims):
        assert (tensor_diagonal.child[i] == tensor.child[i][i]).all()


#
# ######################### ADD ############################
#
# MADHAVA: this needs fixing
@pytest.mark.xfail
def test_add(x: Tensor) -> None:
    z = x + x
    assert isinstance(z, Tensor), "Add: Result is not a Tensor"
    assert (
        z.child.min_vals == 2 * x.child.min_vals
    ).all(), "(Add, Minval) Result is not correct"
    assert (
        z.child.max_vals == 2 * x.child.max_vals
    ).all(), "(Add, Maxval) Result is not correct"


# MADHAVA: this needs fixing
# @pytest.mark.xfail
# def test_single_entity_phi_tensor_serde(x: Tensor) -> None:

#     blob = serialize(x.child)
#     x2 = deserialize(blob)

#     assert (x.child.min_vals == x2.min_vals).all()
#     assert (x.child.max_vals == x2.max_vals).all()


# def test_add(x,y):
#     z = x+y
#     assert isinstance(z, Tensor), "Add: Result is not a Tensor"
#     assert z.child.min_vals == x.child.min_vals + y.child.min_vals, "(Add, Minval) Result is not correct"
#     assert z.child.max_vals == x.child.max_vals + y.child.max_vals, "(Add, Maxval) Result is not correct"
#
# ######################### SUB ############################
#
# def test_sub(x):
#     z=x-x
#     assert isinstance(z, Tensor), "Sub: Result is not a Tensor"
#     assert z.child.min_vals == 0 * x.child.min_vals, "(Sub, Minval) Result is not correct"
#     assert z.child.max_vals == 0 * x.child.max_vals, "(Sub, Maxval) Result is not correct"
#
# def test_sub(x,y):
#     z=x-y
#     assert isinstance(z, Tensor), "Sub: Result is not a Tensor"
#     assert z.child.min_vals == x.child.min_vals - y.child.min_vals, "(Sub, Minval) Result is not correct"
#     assert z.child.max_vals == x.child.max_vals - y.child.max_vals, "(Sub, Maxval) Result is not correct"
#
# ######################### MUL ############################
#
# def test_mul(x):
#     z = x*x
#     assert isinstance(z, Tensor), "Mul: Result is not a Tensor"
#     assert z.child.min_vals == x.child.min_vals ** 2, "(Mul, Minval) Result is not correct"
#     assert z.child.max_vals == x.child.max_vals ** 2, "(Mul, Maxval) Result is not correct"
#
# def test_mul(x,y):
#     z = x*y
#     assert isinstance(z, Tensor), "Mul: Result is not a Tensor"
#     assert z.child.min_vals == x.child.min_vals ** 2, "(Mul, Minval) Result is not correct"
#     assert z.child.max_vals == x.child.max_vals ** 2, "(Mul, Maxval) Result is not correct"
