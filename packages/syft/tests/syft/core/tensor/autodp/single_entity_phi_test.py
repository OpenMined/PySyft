# third party
import numpy as np
import pytest

# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.adp.entity import Entity
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor as SEPT
from syft.core.tensor.tensor import Tensor

# ------------------- EQUALITY OPERATORS -----------------------------------------------

# Global constants
ishan = Entity(name="Ishan")
supreme_leader = Entity(name="Trask")
dims = np.random.randint(10) + 1  # Avoid size 0


@pytest.fixture
def upper_bound() -> np.ndarray:
    """This is used to specify the max_vals for a SEPT that is either binary or randomly generated b/w 0-1"""
    max_values = np.ones(dims)
    return max_values


@pytest.fixture
def lower_bound() -> np.ndarray:
    """This is used to specify the min_vals for a SEPT that is either binary or randomly generated b/w 0-1"""
    min_values = np.zeros(dims)
    return min_values


@pytest.fixture
def reference_data() -> np.ndarray:
    """This generates random data to test the equality operators"""
    reference_data = np.random.random((dims, dims))
    return reference_data


@pytest.fixture
def reference_binary_data() -> np.ndarray:
    """Generate binary data to test the equality operators with bools"""
    binary_data = np.random.randint(2, size=(dims, dims))
    return binary_data


def test_eq(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> SEPT:
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
    # Maybe I can compare both of these results with each other?
    return reference_tensor == same_tensor


def test_eq2(
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


#####################################################################################

gonzalo = Entity(name="Gonzalo")


@pytest.fixture(scope="function")
def x() -> Tensor:
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    x = x.private(min_val=-1, max_val=7, entity=gonzalo)
    return x


@pytest.fixture(scope="function")
def y() -> Tensor:
    y = Tensor(np.array([[-1, -2, -3], [-4, -5, -6]]))
    y = y.private(min_val=-7, max_val=1, entity=gonzalo)
    return y


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
