# stdlib
# stdlib
from typing import Dict

# third party
import numpy as np
from numpy.typing import ArrayLike
import pytest

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectArray
from syft.core.tensor.autodp.gamma_tensor import GammaTensor
from syft.core.tensor.autodp.phi_tensor import PhiTensor as PT
from syft.core.tensor.lazy_repeat_array import lazyrepeatarray as lra
from syft.core.tensor.tensor import Tensor


@pytest.fixture
def ishan() -> ArrayLike:
    return np.array(DataSubjectArray(["φhishan"]))


@pytest.fixture
def traskmaster() -> ArrayLike:
    return np.ndarray(DataSubjectArray(["λamdrew"]))


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
def upper_bound(reference_data: np.ndarray, highest: int) -> lra:
    """This is used to specify the max_vals that is either binary or randomly generated b/w 0-1"""
    return lra(data=highest, shape=reference_data.shape)


@pytest.fixture
def lower_bound(reference_data: np.ndarray, highest: int) -> lra:
    """This is used to specify the min_vals that is either binary or randomly generated b/w 0-1"""
    return lra(data=-highest, shape=reference_data.shape)


@pytest.fixture
def reference_binary_data(dims: int) -> np.ndarray:
    """Generate binary data to test the equality operators with bools"""
    binary_data = np.random.randint(2, size=(dims, dims))
    return binary_data


def test_pos(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    output = +reference_tensor

    assert isinstance(output, PT)
    assert (output.child == reference_tensor.child).all()
    assert (output.min_vals == reference_tensor.min_vals).all()
    assert (output.max_vals == reference_tensor.max_vals).all()
    assert (output.data_subjects == reference_tensor.data_subjects).all()


def test_eq(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Test equality between two identical PhiTensors"""
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    # Duplicate the tensor and check if equality holds
    same_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    assert (
        reference_tensor == same_tensor
    ).all(), "Equality between identical PTs fails"


def test_add_wrong_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Ensure that addition with incorrect types aren't supported"""
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    with pytest.raises(NotImplementedError):
        reference_tensor + "some string"
        reference_tensor + dict()
        # TODO: Double check how tuples behave during addition/subtraction with np.ndarrays


def test_add_tensor_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
    highest: int,
    dims: int,
) -> None:
    """Test addition of a PT with various other kinds of Tensors"""
    ishan = np.broadcast_to(ishan, reference_data.shape)
    # TODO: Add tests for GammaTensor, etc when those are built out.
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    simple_tensor = Tensor(
        child=np.random.randint(
            low=-highest, high=highest, size=(dims + 10, dims + 10), dtype=np.int64
        )
    )

    with pytest.raises(NotImplementedError):
        result = reference_tensor + simple_tensor
        assert isinstance(result, PT), "PT + Tensor != PT"
        assert (
            result.max_vals == reference_tensor.max_vals + simple_tensor.child.max()
        ), "PT + Tensor: incorrect max_vals"
        assert (
            result.min_vals == reference_tensor.min_vals + simple_tensor.child.min()
        ), "PT + Tensor: incorrect min_vals"


def test_add_single_data_subjects(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Test the addition of PhiTensors"""
    ishan = np.broadcast_to(ishan, reference_data.shape)
    tensor1 = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    tensor2 = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    result = tensor2 + tensor1
    # TODO: As we currently convert all operations to gamma tensor,
    # so we include gammatensor for the assert, it should be reverted back to PhiTensor
    assert isinstance(result, (PT, GammaTensor)), "Addition of two PTs is wrong type"
    assert (
        result.max_vals.data == 2 * upper_bound.data
    ).all(), "Addition of two PTs results in incorrect max_vals"
    assert (
        result.min_vals.data == 2 * lower_bound.data
    ).all(), "Addition of two PTs results in incorrect min_vals"

    # Try with negative values
    tensor3 = PT(
        child=reference_data * -1.5,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    result = tensor3 + tensor1
    assert isinstance(result, (PT, GammaTensor)), "Addition of two PTs is wrong type"
    assert (
        result.max_vals == tensor3.max_vals + tensor1.max_vals
    ).all(), "PT + PT results in incorrect max_vals"
    assert (
        result.min_vals == tensor3.min_vals + tensor1.min_vals
    ).all(), "PT + PT results in incorrect min_vals"


def test_serde(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Test basic serde for PT"""
    ishan = np.broadcast_to(ishan, reference_data.shape)
    tensor1 = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    ser = sy.serialize(tensor1)
    de = sy.deserialize(ser)

    assert de == tensor1
    assert (de.child == tensor1.child).all()
    assert (de.min_vals == tensor1.min_vals).all()
    assert (de.max_vals == tensor1.max_vals).all()
    assert (de.data_subjects == tensor1.data_subjects).all()

    assert np.shares_memory(tensor1.child, tensor1.child)
    assert not np.shares_memory(de.child, tensor1.child)


def test_copy(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Test copy for PT"""
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    # Copy the tensor and check if it works
    copy_tensor = reference_tensor.copy()

    assert (reference_tensor == copy_tensor).all(), "Copying of the PT fails"


def test_copy_with(
    reference_data: np.ndarray,
    reference_binary_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Test copy_with for PT"""
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    reference_binary_tensor = PT(
        child=reference_binary_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    # Copy the tensor and check if it works
    copy_with_tensor = reference_tensor.copy_with(reference_data)
    copy_with_binary_tensor = reference_tensor.copy_with(reference_binary_data)

    assert (
        reference_tensor == copy_with_tensor
    ).all(), "Copying of the PT with the given child fails"

    assert (
        reference_binary_tensor == copy_with_binary_tensor
    ).all(), "Copying of the PT with the given child fails"


@pytest.mark.parametrize("kwargs", [{"axis": (1)}])
def test_sum(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
    kwargs: Dict,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    zeros_tensor = PT(
        child=reference_data * 0,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    tensor_sum = tensor.sum(**kwargs)

    assert (tensor_sum.child == reference_data.sum(**kwargs)).all()
    assert zeros_tensor.sum().child == 0


def test_ne_vals(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Test inequality between two different PhiTensors"""
    # TODO: Add tests for GammaTensor when having same values but different entites.
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    comparison_tensor = PT(
        child=reference_data + 1,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    assert (
        reference_tensor != comparison_tensor
    ).all(), "Inequality between different PTs fails"


def test_neg(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Test neg for PT"""
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    neg_tensor = reference_tensor.__neg__()

    assert (neg_tensor.child == reference_tensor.child * -1).all()
    assert (neg_tensor.min_vals == reference_tensor.max_vals * -1).all()
    assert (neg_tensor.max_vals == reference_tensor.min_vals * -1).all()
    assert neg_tensor.shape == reference_tensor.shape


def test_flatten(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    # Default behaviour
    flattened_array = reference_tensor.flatten()

    target_shape = reference_data.flatten().shape
    assert (
        flattened_array.child.shape == target_shape
    ), "Private data not flattened correctly"
    assert (
        flattened_array.min_vals.data == reference_tensor.min_vals.data
    ), "Min vals changed incorrectly in flatten"
    assert (
        flattened_array.max_vals.data == reference_tensor.max_vals.data
    ), "Max vals changed incorrectly in flatten"
    assert flattened_array.min_vals.shape == target_shape
    assert flattened_array.max_vals.shape == target_shape
    assert flattened_array.data_subjects.shape == target_shape
    assert (
        flattened_array.data_subjects == ishan.flatten()
    ).all(), "Data Subjects not flattened properly"


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_add_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor + 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data + 5).all()
    assert output.min_vals.data == reference_tensor.min_vals + 5
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.data == reference_tensor.max_vals + 5
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_sub_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor - 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data - 5).all()
    assert output.min_vals.data == reference_tensor.min_vals - 5
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.data == reference_tensor.max_vals - 5
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_mul_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor * 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data * 5).all()
    assert (output.min_vals.data == reference_tensor.min_vals.data * 5).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == reference_tensor.max_vals.data * 5).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_truediv_public(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor / 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data / 5).all()
    assert (output.min_vals.data == reference_tensor.min_vals.data / 5).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == reference_tensor.max_vals.data / 5).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_add_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    tensor2 = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor + tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data * 2).all()
    assert output.min_vals.data == reference_tensor.min_vals.data * 2
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.data == reference_tensor.max_vals.data * 2
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_sub_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    tensor2 = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor - tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == 0).all()
    assert output.min_vals.data <= output.max_vals.data
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_mul_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    tensor2 = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor * tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data**2).all()
    assert output.min_vals.data <= output.max_vals.data
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_truediv_private(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    tensor2 = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor / tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == 1).all()
    assert output.min_vals.data <= output.max_vals.data
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_eq_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    # Test that it IS equal
    output = reference_tensor == 1
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_ne_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor != 0
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_lt_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor < 2
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_gt_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor > 0
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_le_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor <= 2
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.public_op
def test_ge_public(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor >= 0
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_eq_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    # Test that it IS equal
    output = reference_tensor == reference_tensor.ones_like()
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_ne_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor != reference_tensor.zeros_like()
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_lt_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor < reference_tensor.ones_like() + 5
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_gt_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor + 5 > reference_tensor.zeros_like()
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_le_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor <= reference_tensor.ones_like() + 5
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.equality
@pytest.mark.private_op
def test_ge_private(
    reference_data: np.ndarray,
    upper_bound: lra,
    lower_bound: lra,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.ones_like(reference_data),
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor + 5 >= reference_tensor.zeros_like()
    assert output.shape == reference_tensor.shape
    assert output.child.all()
    assert (output.min_vals.data == 0).all()
    assert output.min_vals.shape == reference_tensor.shape
    assert (output.max_vals.data == 1).all()
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


def test_resize(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:

    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    new_shape = tuple(map(lambda x: x * 2, reference_data.shape))
    resized_tensor = reference_tensor.resize(new_shape)

    no_of_elems = new_shape[0] * new_shape[1] // 4

    flatten_ref = reference_tensor.child.flatten()
    flatten_res = resized_tensor.child.flatten()
    assert (flatten_ref == flatten_res[0:no_of_elems]).all()
    assert (
        flatten_ref == flatten_res[no_of_elems : no_of_elems * 2]  # noqa: E203
    ).all()
    assert (
        flatten_ref == flatten_res[no_of_elems * 2 : no_of_elems * 3]  # noqa: E203
    ).all()
    assert (
        flatten_ref == flatten_res[no_of_elems * 3 : no_of_elems * 4]  # noqa: E203
    ).all()

    assert resized_tensor.min_vals.shape == new_shape
    assert resized_tensor.max_vals.shape == new_shape

    data_subjects_ref = reference_tensor.data_subjects.flatten()
    data_subjects_res = resized_tensor.data_subjects.flatten()
    assert (data_subjects_ref == data_subjects_res[0:no_of_elems]).all()
    assert (
        data_subjects_ref
        == data_subjects_res[no_of_elems : no_of_elems * 2]  # noqa: E203
    ).all()
    assert (
        data_subjects_ref
        == data_subjects_res[no_of_elems * 2 : no_of_elems * 3]  # noqa: E203
    ).all()
    assert (
        data_subjects_ref
        == data_subjects_res[no_of_elems * 3 : no_of_elems * 4]  # noqa: E203
    ).all()


def test_compress(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    condition = list(np.random.choice(a=[False, True], size=(reference_data.shape[0])))
    compressed_tensor = reference_tensor.compress(condition, axis=0)

    new_shape = (
        reference_tensor.shape[0] - len([0 for c in condition if not c]),
        reference_tensor.shape[1],
    )

    comp_ind = 0
    for i, cond in enumerate(condition):
        if cond:
            assert (
                compressed_tensor.child[comp_ind, :] == reference_tensor.child[i, :]
            ).all()
            assert (
                compressed_tensor.data_subjects[comp_ind, :]
                == reference_tensor.data_subjects[i, :]
            ).all()
            comp_ind += 1

    assert compressed_tensor.min_vals.shape == new_shape
    assert compressed_tensor.max_vals.shape == new_shape


def test_squeeze(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.array([reference_data]),
        data_subjects=np.array([ishan]),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    squeezed_tensor = reference_tensor.squeeze()
    assert squeezed_tensor.shape == reference_data.shape
    assert (squeezed_tensor.child == reference_data).all()
    assert (squeezed_tensor.data_subjects == ishan).all()
