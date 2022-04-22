# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.adp.entity import Entity
from syft.core.adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from syft.core.tensor.autodp.ndim_entity_phi import NDimEntityPhiTensor as NDEPT
from syft.core.tensor.tensor import Tensor


@pytest.fixture
def ishan() -> Entity:
    return Entity(name="φhishan")


@pytest.fixture
def traskmaster() -> Entity:
    return Entity(name="λamdrew")


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


def test_eq(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test equality between two identical NDimEntityPhiTensors"""
    reference_tensor = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    # Duplicate the tensor and check if equality holds
    same_tensor = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    assert (
        reference_tensor == same_tensor
    ).all(), "Equality between identical NDEPTs fails"


def test_add_wrong_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Ensure that addition with incorrect types aren't supported"""
    reference_tensor = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    with pytest.raises(NotImplementedError):
        reference_tensor + "some string"
        reference_tensor + dict()
        # TODO: Double check how tuples behave during addition/subtraction with np.ndarrays


def test_add_tensor_types(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    highest: int,
    dims: int,
) -> None:
    """Test addition of a NDEPT with various other kinds of Tensors"""
    # TODO: Add tests for GammaTensor, etc when those are built out.
    reference_tensor = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    simple_tensor = Tensor(
        child=np.random.randint(
            low=-highest, high=highest, size=(dims + 10, dims + 10), dtype=np.int64
        )
    )

    with pytest.raises(NotImplementedError):
        result = reference_tensor + simple_tensor
        assert isinstance(result, NDEPT), "NDEPT + Tensor != NDEPT"
        assert (
            result.max_vals == reference_tensor.max_vals + simple_tensor.child.max()
        ), "NDEPT + Tensor: incorrect max_val"
        assert (
            result.min_vals == reference_tensor.min_vals + simple_tensor.child.min()
        ), "NDEPT + Tensor: incorrect min_val"


def test_add_single_entities(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test the addition of SEPTs"""
    tensor1 = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    tensor2 = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    result = tensor2 + tensor1
    assert isinstance(result, NDEPT), "Addition of two NDEPTs is wrong type"
    assert (
        result.max_vals == 2 * upper_bound
    ).all(), "Addition of two NDEPTs results in incorrect max_val"
    assert (
        result.min_vals == 2 * lower_bound
    ).all(), "Addition of two NDEPTs results in incorrect min_val"

    # Try with negative values
    tensor3 = NDEPT(
        child=reference_data * -1.5,
        entities=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    result = tensor3 + tensor1
    assert isinstance(result, NDEPT), "Addition of two NDEPTs is wrong type"
    assert (
        result.max_vals == tensor3.max_vals + tensor1.max_vals
    ).all(), "NDEPT + NDEPT results in incorrect max_val"
    assert (
        result.min_vals == tensor3.min_vals + tensor1.min_vals
    ).all(), "NDEPT + NDEPT results in incorrect min_val"


def test_serde(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test basic serde for NDEPT"""
    tensor1 = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    ser = sy.serialize(tensor1)
    de = sy.deserialize(ser)

    assert de == tensor1
    assert (de.child == tensor1.child).all()
    assert (de.min_vals == tensor1.min_vals).all()
    assert (de.max_vals == tensor1.max_vals).all()
    assert de.entities == tensor1.entities

    assert np.shares_memory(tensor1.child.child, tensor1.child.child)
    assert not np.shares_memory(de.child.child, tensor1.child.child)


def test_copy(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test copy for NDEPT"""
    reference_tensor = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    # Copy the tensor and check if it works
    copy_tensor = reference_tensor.copy()

    assert (reference_tensor == copy_tensor).all(), "Copying of the NDEPT fails"


def test_copy_with(
    reference_data: np.ndarray,
    reference_binary_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test copy_with for NDEPT"""
    reference_tensor = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    reference_binary_tensor = NDEPT(
        child=reference_binary_data,
        entities=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    encode_func = reference_tensor.child.encode

    # Copy the tensor and check if it works
    copy_with_tensor = reference_tensor.copy_with(encode_func(reference_data))
    copy_with_binary_tensor = reference_tensor.copy_with(
        encode_func(reference_binary_data)
    )

    assert (
        reference_tensor == copy_with_tensor
    ).all(), "Copying of the NDEPT with the given child fails"

    assert (
        reference_binary_tensor == copy_with_binary_tensor
    ).all(), "Copying of the NDEPT with the given child fails"


def test_sum(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    dims: int,
) -> None:
    zeros_tensor = NDEPT(
        child=reference_data * 0,
        entities=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )
    tensor = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    encode_func = tensor.child.encode
    tensor_sum = tensor.sum()

    assert tensor_sum.child.child == encode_func(reference_data).sum()
    assert zeros_tensor.sum().child.child == 0


def test_ne_vals(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test inequality between two different NDimEntityPhiTensors"""
    # TODO: Add tests for GammaTensor when having same values but different entites.
    reference_tensor = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    comparison_tensor = NDEPT(
        child=reference_data + 1,
        entities=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
    )

    assert (
        reference_tensor != comparison_tensor
    ).all(), "Inequality between different NDEPTs fails"


def test_neg(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test neg for NDEPT"""
    reference_tensor = NDEPT(
        child=reference_data, entities=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    neg_tensor = reference_tensor.__neg__()

    assert (neg_tensor.child == reference_tensor.child * -1).all()
    assert (neg_tensor.min_vals == reference_tensor.max_vals * -1).all()
    assert (neg_tensor.max_vals == reference_tensor.min_vals * -1).all()
    assert neg_tensor.shape == reference_tensor.shape
