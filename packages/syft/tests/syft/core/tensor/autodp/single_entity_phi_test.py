# stdlib
from random import randint
from random import sample

# third party
import numpy as np
import pytest

# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.adp.entity import Entity
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor as SEPT
from syft.core.tensor.tensor import Tensor

# Global constants
ishan = Entity(name="Ishan")
supreme_leader = Entity(name="Trask")
dims = np.random.randint(10) + 3  # Avoid size 0
high = 50


@pytest.fixture
def reference_data() -> np.ndarray:
    """This generates random data to test the equality operators"""
    reference_data = np.random.randint(
        low=-high, high=high, size=(dims, dims), dtype=np.int32
    )
    return reference_data


@pytest.fixture
def upper_bound(reference_data: np.ndarray) -> np.ndarray:
    """This is used to specify the max_vals for a SEPT that is either binary or randomly generated b/w 0-1"""
    max_values = np.ones_like(reference_data) * high
    return max_values


@pytest.fixture
def lower_bound(reference_data: np.ndarray) -> np.ndarray:
    """This is used to specify the min_vals for a SEPT that is either binary or randomly generated b/w 0-1"""
    min_values = np.ones_like(reference_data) * -high
    return min_values


@pytest.fixture
def reference_binary_data() -> np.ndarray:
    """Generate binary data to test the equality operators with bools"""
    binary_data = np.random.randint(2, size=(dims, dims))
    return binary_data


@pytest.mark.skip(
    reason="Test passes, but to check the test throws a Deprecation Warning for .all()"
)
def test_eq(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> SEPT:
    """Test equality between Private Tensors with different owners. This is currently not implemented."""
    tensor1 = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    tensor2 = SEPT(
        child=reference_data,
        entity=supreme_leader,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    with pytest.raises(NotImplementedError):
        return tensor1 == tensor2


def test_eq_ndarray(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
    reference_binary_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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


def test_ne_shapes(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> None:
    """Test non-equality between SEPTs with different shapes"""
    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    comparison_tensor = SEPT(
        child=np.random.randint(
            low=-high, high=high, size=(dims + 10, dims + 10), dtype=np.int32
        ),
        entity=ishan,
        max_vals=np.ones(dims + 10),
        min_vals=np.ones(dims + 10),
    )

    with pytest.raises(Exception):
        reference_tensor != comparison_tensor
    return None


def test_ne_broadcastability(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> None:
    """Test non-equality between SEPTs of different entities"""
    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    comparison_tensor = SEPT(
        child=reference_data,
        entity=supreme_leader,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    with pytest.raises(NotImplementedError):
        reference_tensor != comparison_tensor
    return None


def test_add_wrong_types(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> None:
    """Test addition of a SEPT with various other kinds of Tensors"""
    # TODO: Add tests for REPT, GammaTensor, etc when those are built out.

    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    simple_tensor = Tensor(
        child=np.random.randint(
            low=-high, high=high, size=(dims + 10, dims + 10), dtype=np.int32
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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


def test_add_diff_entities(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> None:
    """Test the addition of SEPTs"""

    tensor1 = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    tensor2 = SEPT(
        child=reference_data,
        entity=supreme_leader,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    assert tensor2.entity != tensor1.entity, "Entities aren't actually different"

    with pytest.raises(NotImplementedError):
        tensor2 + tensor1
    return None


def test_add_sub_equivalence(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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


def test_transpose_simple_types() -> None:
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
    assert int_tensor_transposed == int_tensor, "Transpose: equality error"

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
    assert float_tensor_transposed == float_tensor, "Transpose: equality error"

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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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


def test_transpose_non_square_matrix() -> None:
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
        low=-high, high=high, size=(rows, cols), dtype=np.int32
    )
    tensor = SEPT(
        child=non_square_data,
        entity=ishan,
        max_vals=np.ones_like(non_square_data) * high,
        min_vals=np.ones_like(non_square_data) * -high,
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> None:
    """Ensure reshape happens when it is able"""
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    new_shape = reference_data.flatten().shape[0]
    reference_tensor.reshape(new_shape)


def test_reshape_fail(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> None:
    """Ensure resize happens when it is able"""
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    new_shape = reference_data.flatten().shape[0]
    reference_tensor.reshape(new_shape)


def test_resize_fail(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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


def test_squeeze() -> None:
    """Test that squeeze works on an ideal case"""
    _data = np.random.randint(
        low=-high, high=high, size=(10, 1, 10, 1, 10), dtype=np.int32
    )
    initial_shape = _data.shape

    reference_tensor = SEPT(
        child=_data,
        max_vals=np.ones_like(_data) * high,
        min_vals=np.ones_like(_data) * -high,
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


def test_squeeze_correct_axes() -> None:
    """Test that squeeze works on an ideal case with correct axes specified"""
    _data = np.random.randint(
        low=-high, high=high, size=(10, 1, 10, 1, 10), dtype=np.int32
    )
    initial_shape = _data.shape

    reference_tensor = SEPT(
        child=_data,
        max_vals=np.ones_like(_data) * high,
        min_vals=np.ones_like(_data) * -high,
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


def test_swap_axes() -> None:
    """Test that swap_axes works on an ideal case"""
    data = np.random.randint(
        low=-high, high=high, size=(10, 1, 10, 1, 10), dtype=np.int32
    )
    initial_shape = data.shape

    reference_tensor = SEPT(
        child=data,
        max_vals=np.ones_like(data) * high,
        min_vals=np.ones_like(data) * -high,
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


def test_compress(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
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


def test_partition(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> None:
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    k = 1

    reference_tensor.partition(k)
    assert reference_tensor != reference_data, "Partition did not work as expected"
    reference_data.partition(k)
    assert reference_tensor == reference_data, "Partition did not work as expected"


def test_partition_axis(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> None:
    reference_tensor = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )

    k = 1

    reference_tensor.partition(k, axis=1)
    assert reference_tensor != reference_data, "Partition did not work as expected"
    reference_data.partition(k, axis=1)
    assert reference_tensor == reference_data, "Partition did not work as expected"


# End of Ishan's tests


gonzalo = Entity(name="Gonzalo")


@pytest.fixture(scope="function")
def x() -> Tensor:
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    x = x.private(min_val=-1, max_val=7, entities=[gonzalo])
    return x


@pytest.fixture(scope="function")
def y() -> Tensor:
    y = Tensor(np.array([[-1, -2, -3], [-4, -5, -6]]))
    y = y.private(min_val=-7, max_val=1, entities=[gonzalo])
    return y


ent = Entity(name="test")
ent2 = Entity(name="test2")

dims = np.random.randint(10) + 1

child1 = np.random.randint(low=-2, high=4, size=dims)
upper1 = np.full(dims, 3, dtype=np.int32)
low1 = np.full(dims, -2, dtype=np.int32)
child2 = np.random.randint(low=4, high=7, size=dims)
upper2 = np.full(dims, 6, dtype=np.int32)
low2 = np.full(dims, 4, dtype=np.int32)

tensor1 = SEPT(child=child1, entity=ent, max_vals=upper1, min_vals=low1)
# same entity, same data
tensor2 = SEPT(child=child1, entity=ent, max_vals=upper1, min_vals=low1)
# same entity, different data
tensor3 = SEPT(child=child2, entity=ent, max_vals=upper2, min_vals=low2)
# different entity, same data
tensor4 = SEPT(child=child1, entity=ent2, max_vals=upper1, min_vals=low1)
# different entity, different data
tensor5 = SEPT(child=child2, entity=ent2, max_vals=upper2, min_vals=low2)


# helper function to replace the value at a given index of a single entity phi tensor
def change_elem(tensor, ind, val) -> SingleEntityPhiTensor:
    tensor.child[ind] = val
    return tensor


simple_type1 = randint(-6, -4)
simple_type2 = randint(4, 6)


def test_le() -> None:

    assert tensor1.__le__(tensor2).child.all()
    assert not tensor3.__le__(tensor1).child.all()
    assert tensor1.__le__(tensor4) == NotImplemented
    assert tensor1.__le__(tensor5) == NotImplemented
    assert not tensor1.__le__(simple_type1).child.all()
    assert tensor1.__le__(simple_type2).child.all()


def test_ge() -> None:

    assert tensor1.__ge__(tensor2).child.all()
    assert not tensor1.__ge__(tensor3).child.all()
    assert tensor1.__ge__(tensor4) == NotImplemented
    assert tensor1.__ge__(tensor5) == NotImplemented
    assert tensor1.__ge__(simple_type1).child.all()
    assert not tensor1.__ge__(simple_type2).child.all()


def test_lt() -> None:

    assert not tensor1.__lt__(tensor2).child.all()
    assert tensor1.__lt__(tensor3).child.all()
    assert tensor1.__lt__(tensor4) == NotImplemented
    assert tensor1.__lt__(tensor5) == NotImplemented
    assert not tensor1.__lt__(simple_type1).child.all()
    assert tensor1.__lt__(simple_type2)


def test_gt() -> None:

    assert not tensor1.__gt__(tensor2).child.all()
    assert not tensor1.__gt__(tensor3).child.all()
    assert tensor1.__gt__(tensor4) == NotImplemented
    assert tensor1.__gt__(tensor5) == NotImplemented
    assert tensor1.__gt__(simple_type1).child.all()
    assert not tensor1.__gt__(simple_type2).child.all()


rand1 = np.random.randint(-4, 1)
rand2 = np.random.randint(1, 5)
clipped_tensor1 = tensor1.clip(rand1, rand2).child
clipped_tensor2 = tensor1.clip(rand2, rand1).child
clipped_tensor3 = tensor1.clip(rand1, None).child


def test_clip() -> None:
    assert ((clipped_tensor1 >= rand1) & (clipped_tensor1 <= rand2)).all()
    assert (clipped_tensor2 == rand1).all()
    assert (clipped_tensor3 >= rand1).all()


tensor1_copy = tensor1.copy()
tensor3_copy = tensor3.copy()


def test_copy() -> None:
    assert (
        (tensor1_copy.child == tensor1.child).all()
        & (tensor1_copy.min_vals == tensor1.min_vals).all()
        & (tensor1_copy.max_vals == tensor1.max_vals).all()
    )
    assert not (tensor3_copy.child == change_elem(tensor3, 0, rand1).child).all()


indices = sample(range(dims), dims)
tensor1_take = tensor1.take(indices)


def test_take() -> None:
    for i in range(dims):
        assert tensor1_take.child[i] == tensor1.child[indices[i]]


tensor6 = SingleEntityPhiTensor(
    child=np.arange(dims * dims).reshape(dims, dims),
    entity=ent,
    max_vals=np.full((dims, dims), dims ** 2 - 1),
    min_vals=np.full((dims, dims), 0),
)
tensor6_diagonal = tensor6.diagonal()


def test_diagonal() -> None:
    for i in range(dims):
        assert tensor6_diagonal.child[i] == tensor6.child[i][i]


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
@pytest.mark.xfail
def test_single_entity_phi_tensor_serde(x: Tensor) -> None:

    blob = serialize(x.child)
    x2 = deserialize(blob)

    assert (x.child.min_vals == x2.min_vals).all()
    assert (x.child.max_vals == x2.max_vals).all()


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
