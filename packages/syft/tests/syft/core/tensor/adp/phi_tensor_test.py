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
def dsa(dims: int) -> DataSubjectArray:
    return DataSubjectArray.from_objs(np.ones((dims, dims)))


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
        reference_tensor.child == same_tensor.child
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

    assert (
        reference_tensor.child == copy_tensor.child
    ).all(), "Copying of the PT fails"


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
        reference_tensor.child == copy_with_tensor.child
    ).all(), "Copying of the PT with the given child fails"

    assert (
        reference_binary_tensor.child == copy_with_binary_tensor.child
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


def test_pow(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=np.broadcast_to(ishan, reference_data.shape),
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor.__pow__(2)
    assert (result.child == (reference_data**2)).all()
    assert result.child.min() >= result.min_vals.data
    assert result.child.max() <= result.max_vals.data


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
        reference_tensor.child != comparison_tensor.child
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
    resized_tensor = reference_tensor.resize(new_shape, refcheck=False)

    flatten_ref = reference_tensor.child.flatten()
    flatten_res = resized_tensor.child.flatten()

    assert (flatten_ref == flatten_res).all()

    assert resized_tensor.min_vals.shape == new_shape
    assert resized_tensor.max_vals.shape == new_shape

    data_subjects_ref = reference_tensor.data_subjects.flatten()
    data_subjects_res = resized_tensor.data_subjects.flatten()

    assert (data_subjects_ref == data_subjects_res).all()


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
    # if we have all False compress throws an exception because the size of the slices is 0
    while not any(condition):
        condition = list(
            np.random.choice(a=[False, True], size=(reference_data.shape[0]))
        )

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


def test_any(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=np.array(ishan),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    result = (reference_tensor == reference_data).any()
    assert result.child
    assert result.data_subjects.shape == ()
    assert (result.data_subjects == ishan).any()

    result = (reference_tensor == reference_data).any(axis=0)
    assert result.shape == (reference_data.shape[0],)
    assert result.data_subjects.shape == (reference_data.shape[0],)
    assert (result.data_subjects == ishan).any()

    result = (reference_tensor == reference_data).any(keepdims=True)
    assert result.shape == (1, 1)
    assert result.data_subjects.shape == (1, 1)
    assert (result.data_subjects == ishan).any()

    result = (reference_tensor == reference_data).any(keepdims=True, axis=0)
    assert result.shape == (1, reference_tensor.shape[0])
    assert result.data_subjects.shape == (1, reference_tensor.shape[0])
    assert (result.data_subjects == ishan).any()

    condition = list(
        np.random.choice(a=[False, True], size=(reference_data.shape[0] - 1))
    )
    condition.append(
        True
    )  # If condition = [False, False, False ... False], this test will fail
    result = (reference_tensor == reference_data).any(where=condition)
    assert result.child
    assert result.data_subjects.shape == ()


def test_all(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.array(reference_data),
        data_subjects=np.array(ishan),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    result = (reference_tensor == reference_data).all()
    assert result.child
    assert result.data_subjects.shape == ()
    assert (result.data_subjects == ishan).all()

    result = (reference_tensor == reference_data).all(axis=0)
    assert result.shape == (reference_data.shape[0],)
    assert result.data_subjects.shape == (reference_data.shape[0],)
    assert (result.data_subjects == ishan).all()

    result = (reference_tensor == reference_data).all(keepdims=True)
    assert result.shape == (1, 1)
    assert result.data_subjects.shape == (1, 1)
    assert (result.data_subjects == ishan).all()

    result = (reference_tensor == reference_data).all(keepdims=True, axis=0)
    assert result.shape == (1, reference_tensor.shape[0])
    assert result.data_subjects.shape == (1, reference_tensor.shape[0])
    assert (result.data_subjects == ishan).all()

    condition = list(
        np.random.choice(a=[False, True], size=(reference_data.shape[0] - 1))
    )
    condition.append(True)
    result = (reference_tensor == reference_data).all(where=condition)
    assert (
        result.child
    )  # If condition = [False, False, False ... False], this test will fail
    assert result.data_subjects.shape == ()


def test_and_wrong_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Ensure that and with incorrect types aren't supported"""
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    with pytest.raises(NotImplementedError):
        reference_tensor & "some string"
        reference_tensor & dict()


def test_and_tensor_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
    highest: int,
    dims: int,
) -> None:
    """Test and of a PT with various other kinds of Tensors"""
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
        result = reference_tensor & simple_tensor
        assert isinstance(result, PT), "PT & Tensor != PT"
        assert result.max_vals == 1, "PT & Tensor: incorrect max_vals"
        assert result.min_vals == 0, "PT & Tensor: incorrect min_vals"


def test_and_single_data_subjects(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Test the mod of PhiTensors"""
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

    result = tensor2 & tensor1
    # TODO: As we currently convert all operations to gamma tensor,
    # so we include gammatensor for the assert, it should be reverted back to PhiTensor
    assert isinstance(result, (PT, GammaTensor)), "And of two PTs is wrong type"
    assert (
        result.child.min() >= result.min_vals.data
    ), "And of two PTs results in incorrect max_vals"
    assert (
        result.child.max() <= result.max_vals.data
    ), "And of two PTs results in incorrect min_vals"


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_and_public(
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

    output = reference_tensor & 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data & 5).all()
    assert output.child.min() >= output.min_vals.data
    assert output.min_vals.shape == reference_tensor.shape
    assert output.child.max() <= output.max_vals.data
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()

    output = reference_tensor & -5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data & -5).all()
    assert output.child.min() >= output.min_vals.data
    assert output.min_vals.shape == reference_tensor.shape
    assert output.child.max() <= output.max_vals.data
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_and_private(
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

    dims = reference_data.shape[0]
    new_reference_data = np.random.randint(
        low=lower_bound.data, high=upper_bound.data, size=(dims, dims), dtype=np.int32
    )
    tensor2 = PT(
        child=new_reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor & tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data & new_reference_data).all()
    assert output.child.min() >= output.min_vals.data
    assert output.min_vals.shape == reference_tensor.shape
    assert output.child.max() <= output.max_vals.data
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


def test_or_wrong_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Ensure that or with incorrect types aren't supported"""
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    with pytest.raises(NotImplementedError):
        reference_tensor | "some string"
        reference_tensor | dict()


def test_or_tensor_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
    highest: int,
    dims: int,
) -> None:
    """Test or of a PT with various other kinds of Tensors"""
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
        result = reference_tensor | simple_tensor
        assert isinstance(result, PT), "PT | Tensor != PT"
        assert result.max_vals == 1, "PT | Tensor: incorrect max_vals"
        assert result.min_vals == 0, "PT | Tensor: incorrect min_vals"


def test_or_single_data_subjects(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Test the or of PhiTensors"""
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

    result = tensor2 | tensor1
    # TODO: As we currently convert all operations to gamma tensor,
    # so we include gammatensor for the assert, it should be reverted back to PhiTensor
    assert isinstance(result, (PT, GammaTensor)), "Or of two PTs is wrong type"
    assert (
        result.child.max() <= result.max_vals.data
    ), "Or of two PTs results in incorrect max_vals"
    assert (
        result.child.min() >= result.min_vals.data
    ), "Or of two PTs results in incorrect min_vals"


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_or_public(
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

    output = reference_tensor | 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data | 5).all()
    assert output.child.min() >= output.min_vals.data
    assert output.min_vals.shape == reference_tensor.shape
    assert output.child.max() <= output.max_vals.data
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()

    output = reference_tensor | -5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data | -5).all()
    assert output.child.min() >= output.min_vals.data
    assert output.min_vals.shape == reference_tensor.shape
    assert output.child.max() <= output.max_vals.data
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_or_private(
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

    dims = reference_data.shape[0]
    new_reference_data = np.random.randint(
        low=lower_bound.data, high=upper_bound.data, size=(dims, dims), dtype=np.int32
    )
    tensor2 = PT(
        child=new_reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor | tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data | new_reference_data).all()
    assert output.child.min() >= output.min_vals.data
    assert output.min_vals.shape == reference_tensor.shape
    assert output.child.max() <= output.max_vals.data
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


def test_take(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.array(reference_data),
        data_subjects=np.array(ishan),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    indices = [2]
    result = reference_tensor.take(indices, axis=0)
    assert (result.child == reference_tensor.child[indices, :]).all()
    assert (result.min_vals == reference_tensor.min_vals[indices, :]).all()
    assert (result.max_vals == reference_tensor.max_vals[indices, :]).all()
    assert (result.data_subjects == reference_tensor.data_subjects[indices, :]).all()


def test_put(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.array(reference_data),
        data_subjects=np.array(ishan),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    no_values = reference_tensor.shape[0]
    new_values = np.random.randint(low=-50, high=50, size=(no_values), dtype=np.int32)
    indices = np.random.randint(
        low=0, high=no_values * no_values - no_values - 1, size=(1), dtype=np.int32
    )[0]

    result = reference_tensor.put(range(indices, indices + no_values), new_values)
    flatten_results = result.child.flat[indices:]
    assert (flatten_results[:no_values] == new_values).all()


def test_abs(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.array(reference_data),
        data_subjects=np.array(ishan),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    result = abs(reference_tensor)
    assert (result.child == abs(reference_tensor.child)).all()
    assert (result.min_vals.data >= 0).all()
    assert (result.max_vals.data >= 0).all()
    assert (result.data_subjects == reference_tensor.data_subjects).all()


def test_argmax(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.array(reference_data),
        data_subjects=np.array(ishan),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    result = reference_tensor.argmax()
    reference_result = reference_tensor.child.argmax()
    assert (result.child == reference_result).all()
    assert (
        result.data_subjects == reference_tensor.data_subjects.item(reference_result)
    ).all()

    result = reference_tensor.argmax(axis=0)
    reference_result = reference_tensor.child.argmax(axis=0)
    assert (result.child == reference_result).all()
    assert (
        result.data_subjects == reference_tensor.data_subjects[reference_result]
    ).all()


def test_swapaxes(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.array(reference_data),
        data_subjects=np.array(ishan),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    result = reference_tensor.swapaxes(0, 1)
    reference_result = reference_tensor.child.swapaxes(0, 1)
    assert (result.child == reference_result).all()
    assert (result.data_subjects == reference_tensor.data_subjects.swapaxes(0, 1)).all()
    assert result.min_vals.shape == reference_result.shape
    assert result.max_vals.shape == reference_result.shape


def test_argmin(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.array(reference_data),
        data_subjects=np.array(ishan),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    result = reference_tensor.argmin()
    reference_result = reference_tensor.child.argmin()
    assert (result.child == reference_result).all()
    assert (
        result.data_subjects == reference_tensor.data_subjects.item(reference_result)
    ).all()

    result = reference_tensor.argmin(axis=0)
    reference_result = reference_tensor.child.argmin(axis=0)
    assert (result.child == reference_result).all()
    assert (
        result.data_subjects == reference_tensor.data_subjects[reference_result]
    ).all()


def test_ptp(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.array(reference_data),
        data_subjects=np.array(ishan),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    result = reference_tensor.ptp()
    assert result.child == reference_data.ptp()
    assert (result.data_subjects == ishan).any()
    assert result.min_vals.data == 0
    assert result.max_vals.data == upper_bound - lower_bound

    result = reference_tensor.ptp(axis=0)
    assert (result.child == reference_data.ptp(axis=0, keepdims=True)).all()
    assert (result.data_subjects == ishan).any()
    assert result.min_vals.data == 0
    assert result.max_vals.data == upper_bound - lower_bound


def test_mod_wrong_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Ensure that mod with incorrect types aren't supported"""
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    with pytest.raises(NotImplementedError):
        reference_tensor % "some string"
        reference_tensor % dict()
        # TODO: Double check how tuples behave during addition/subtraction with np.ndarrays


def test_mod_tensor_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
    highest: int,
    dims: int,
) -> None:
    """Test mod of a PT with various other kinds of Tensors"""
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
        result = reference_tensor % simple_tensor
        assert isinstance(result, PT), "PT % Tensor != PT"
        assert (
            result.max_vals == simple_tensor.child.max()
        ), "PT % Tensor: incorrect max_vals"
        assert result.min_vals == 0, "PT % Tensor: incorrect min_vals"


def test_mod_single_data_subjects(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    """Test the mod of PhiTensors"""
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

    result = tensor2 % tensor1
    # TODO: As we currently convert all operations to gamma tensor,
    # so we include gammatensor for the assert, it should be reverted back to PhiTensor
    assert isinstance(result, (PT, GammaTensor)), "Addition of two PTs is wrong type"
    assert result.max_vals.data >= max(
        0, tensor1.child.max()
    ), "Mod of two PTs results in incorrect max_vals"
    assert result.min_vals.data <= min(
        0, tensor1.child.min()
    ), "Mod of two PTs results in incorrect min_vals"

    # Try with negative values
    tensor3 = PT(
        child=reference_data * -1.5,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    result = tensor3 % tensor1
    assert isinstance(result, (PT, GammaTensor)), "Mod of two PTs is wrong type"
    assert result.max_vals.data >= max(
        0, tensor1.child.max()
    ), "PT % PT results in incorrect max_vals"
    assert result.min_vals.data <= min(
        0, tensor1.child.min()
    ), "PT % PT results in incorrect min_vals"


@pytest.mark.arithmetic
@pytest.mark.public_op
def test_mod_public(
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

    output = reference_tensor % 5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data % 5).all()
    assert output.min_vals.data == 0
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.data == 5
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()

    output = reference_tensor % -5
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data % -5).all()
    assert output.min_vals.data == -5
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.data == 0
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


@pytest.mark.arithmetic
@pytest.mark.private_op
def test_mod_private(
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

    dims = reference_data.shape[0]
    new_reference_data = np.random.randint(
        low=lower_bound.data, high=upper_bound.data, size=(dims, dims), dtype=np.int32
    )
    tensor2 = PT(
        child=new_reference_data,
        data_subjects=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    output = reference_tensor % tensor2
    assert output.shape == reference_tensor.shape
    assert (output.child == reference_data % new_reference_data).all()
    assert output.min_vals.data <= min(0, reference_data.min())
    assert output.min_vals.shape == reference_tensor.shape
    assert output.max_vals.data >= max(0, reference_data.max())
    assert output.max_vals.shape == reference_tensor.shape
    assert (output.data_subjects == reference_tensor.data_subjects).all()


def test_nonzero(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: DataSubjectArray,
) -> None:
    ishan = np.broadcast_to(ishan, reference_data.shape)
    reference_tensor = PT(
        child=np.array(reference_data),
        data_subjects=np.array(ishan),
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    result = reference_tensor.nonzero()
    reference_result = np.array(reference_tensor.child.nonzero())
    assert (result.child == reference_result).all()
    assert (
        result.data_subjects
        == reference_tensor.data_subjects[reference_tensor.child != 0]
    ).all()
    assert result.min_vals.shape == reference_result.shape
    assert result.max_vals.shape == reference_result.shape


def test_cumsum(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    result = tensor.cumsum()
    assert (result.child == reference_data.cumsum()).all()
    assert (result.child >= result.min_vals.data).all()
    assert (result.child <= result.max_vals.data).all()

    result = tensor.cumsum(axis=1)
    assert (result.child == reference_data.cumsum(axis=1)).all()
    assert (result.child >= result.min_vals.data).all()
    assert (result.child <= result.max_vals.data).all()


def test_var(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor.var()
    assert result.child == reference_data.var()
    assert result.child >= result.min_vals.data
    assert result.child <= result.max_vals.data

    result = tensor.var(axis=1)
    assert (result.child == reference_data.var(axis=1)).all()
    assert (result.child >= result.min_vals.data).all()
    assert (result.child <= result.max_vals.data).all()


def test_prod(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor.prod()
    assert result.child == reference_data.prod()
    # assert result.child >= result.min_vals.data
    # assert result.child <= result.max_vals.data

    result = tensor.prod(axis=1)
    assert (result.child == reference_data.prod(axis=1)).all()
    # assert result.child >= result.min_vals.data
    # assert result.child <= result.max_vals.data


def test_std(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor.std()
    assert result.child == reference_data.std()
    assert result.child >= result.min_vals.data
    assert result.child <= result.max_vals.data

    result = tensor.std(axis=1)
    assert (result.child == reference_data.std(axis=1)).all()
    assert (result.child >= result.min_vals.data).all()
    assert (result.child <= result.max_vals.data).all()


def test_cumprod(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    # Note: It's difficult to test the min/max values for cumprod because of the extremely high bounds this op gives.
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    result = tensor.cumprod()
    assert (result.child == reference_data.cumprod()).all()
    # assert (result.child >= result.min_vals.data).all()
    # assert (result.child <= result.max_vals.data).all()

    result = tensor.cumprod(axis=1)
    assert (result.child == reference_data.cumprod(axis=1)).all()
    # assert (result.child >= result.min_vals.data).all()
    # assert (result.child <= result.max_vals.data).all()


def test_floordiv(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
    dims: int,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor // 5
    assert (result.child == (reference_data // 5)).all()
    assert (result.child >= result.min_vals.data).all()
    assert (result.child <= result.max_vals.data).all()

    tensor2 = PT(
        child=reference_data + 1,
        data_subjects=dsa,
        min_vals=lower_bound + 1,
        max_vals=upper_bound + 1,
    )

    result = tensor // tensor2
    assert (result.child == (reference_data // (reference_data + 1))).all()
    assert (result.child.min() >= result.min_vals.data).all()
    # assert (result.child.max() <= result.max_vals.data).all()  # Currently flaky for some reason

    array = np.ones((dims, dims))

    result = tensor // array
    assert (result.child == (reference_data // array)).all()
    assert (result.child >= result.min_vals.data).all()
    assert (result.child <= result.max_vals.data).all()


def test_trace(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor.trace()
    assert result.child == reference_data.trace()
    assert result.child >= result.min_vals.data
    assert result.child <= result.max_vals.data

    result = tensor.trace(offset=1)
    assert result.child == reference_data.trace(offset=1)
    assert result.child >= result.min_vals.data
    assert result.child <= result.max_vals.data


def test_max(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor.max()
    assert result.child == reference_data.max()
    assert result.child >= result.min_vals.data
    assert result.child <= result.max_vals.data


def test_min(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor.min()
    assert result.child == reference_data.min()
    assert result.child >= result.min_vals.data
    assert result.child <= result.max_vals.data


def test_matmul(
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
    result = reference_tensor @ reference_tensor
    assert (result.child == (reference_data @ reference_data)).all()
    assert (result.child.min() >= result.min_vals.data).all()
    assert (result.child.max() <= result.max_vals.data).all()


def test_xor(
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
    other = np.ones_like(reference_data)
    result = reference_tensor ^ other

    assert (result.child == (reference_data ^ other)).all()
    assert (result.child.max() <= result.max_vals.data).all()
    assert (result.child.max() >= result.min_vals.data).all()

    other = np.zeros_like(reference_data)
    result = reference_tensor ^ other
    assert (result.child == (reference_data ^ other)).all()
    assert (result.child.max() <= result.max_vals.data).all()
    assert (result.child.max() >= result.min_vals.data).all()

    result = reference_tensor ^ reference_tensor
    assert (result.child == (reference_data ^ reference_data)).all()
    assert (result.child.max() <= result.max_vals.data).all()
    assert (result.child.max() >= result.min_vals.data).all()

    other = PT(
        child=reference_data,
        data_subjects=DataSubjectArray.from_objs(np.ones_like(reference_data)),
        min_vals=lower_bound,
        max_vals=upper_bound,
    )

    result = reference_tensor ^ other
    assert (result.child == (reference_data ^ other.child)).all()
    assert (result.child.max() <= result.max_vals.data).all()
    assert (result.child.max() >= result.min_vals.data).all()


def test_lshift(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor << 10
    assert (result.child == reference_data << 10).all()
    assert result.child.max() <= result.max_vals.data
    assert result.child.min() >= result.min_vals.data


def test_rshift(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor >> 10
    assert (result.child == reference_data >> 10).all()
    assert result.child.max() <= result.max_vals.data
    assert result.child.min() >= result.min_vals.data


def test_round(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    data = reference_data / 100
    tensor = PT(
        child=data,
        data_subjects=dsa,
        min_vals=lower_bound / 100,
        max_vals=upper_bound / 100,
    )

    result = tensor.round()
    assert (result.child == data.round()).all()
    assert result.child.min() >= result.min_vals.data
    assert result.child.max() <= result.max_vals.data

    result = tensor.round(2)
    assert (result.child == data.round(2)).all()
    assert result.child.min() >= result.min_vals.data
    assert result.child.max() <= result.max_vals.data


def test_sort(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )

    result = tensor.sort()
    data = reference_data.copy()  # make a copy incase this changes the fixture
    data.sort()
    assert (result.child == data).all()
    assert result.child.min() >= result.min_vals.data
    assert result.child.max() <= result.max_vals.data

    result = tensor.sort(axis=1)
    data = reference_data.copy()  # make a copy incase this changes the fixture
    data.sort(axis=1)
    assert (result.child == data).all()
    assert result.child.min() >= result.min_vals.data
    assert result.child.max() <= result.max_vals.data


def test_argsort(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor.argsort()
    assert (result.child == reference_data.argsort()).all()
    assert result.child.min() >= result.min_vals.data
    assert result.child.max() <= result.max_vals.data

    result = tensor.argsort(axis=1)
    assert (result.child == reference_data.argsort(axis=1)).all()
    assert result.child.min() >= result.min_vals.data
    assert result.child.max() <= result.max_vals.data


def test_transpose(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor.transpose()
    assert (result.child == reference_data.transpose()).all()
    assert result.child.min() >= result.min_vals.data
    assert result.child.max() <= result.max_vals.data

    result = tensor.transpose((1, 0))
    assert (result.child == reference_data.transpose((1, 0))).all()
    assert result.child.min() >= result.min_vals.data
    assert result.child.max() <= result.max_vals.data


def test_reshape(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    dsa: DataSubjectArray,
    dims: int,
) -> None:
    tensor = PT(
        child=reference_data,
        data_subjects=dsa,
        min_vals=lower_bound,
        max_vals=upper_bound,
    )
    result = tensor.reshape((1, dims * dims))
    assert (result.child == reference_data.reshape((1, dims * dims))).all()
    assert result.child.min() >= result.min_vals.data
    assert result.child.max() <= result.max_vals.data
