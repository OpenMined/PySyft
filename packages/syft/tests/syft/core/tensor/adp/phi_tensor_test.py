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
def upper_bound(reference_data: np.ndarray, highest: int) -> np.ndarray:
    """This is used to specify the max_vals that is either binary or randomly generated b/w 0-1"""
    max_values = np.ones_like(reference_data) * highest
    return max_values


@pytest.fixture
def lower_bound(reference_data: np.ndarray, highest: int) -> np.ndarray:
    """This is used to specify the min_vals that is either binary or randomly generated b/w 0-1"""
    min_values = np.ones_like(reference_data) * -highest
    return min_values


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
        result.max_vals == 2 * upper_bound
    ).all(), "Addition of two PTs results in incorrect max_vals"
    assert (
        result.min_vals == 2 * lower_bound
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
