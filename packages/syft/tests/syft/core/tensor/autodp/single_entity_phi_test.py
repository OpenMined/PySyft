# stdlib
from random import randint

# third party
import numpy as np
import pytest

# syft absolute
from syft import deserialize
from syft import serialize
from syft.core.adp.entity import Entity
from syft.core.adp.vm_private_scalar_manager import VirtualMachinePrivateScalarManager
from syft.core.tensor.autodp.initial_gamma import IntermediateGammaTensor as IGT
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor as SEPT
from syft.core.tensor.tensor import Tensor

# Global constants
ishan = Entity(name="Ishan")
supreme_leader = Entity(name="Trask")
dims = np.random.randint(10) + 3  # Avoid size 0 or 1
highest = 100


@pytest.fixture
def upper_bound() -> np.ndarray:
    """This is used to specify the max_vals for a SEPT that is either binary or randomly generated b/w 0-1"""
    max_values = np.ones(dims, dtype=int) * highest
    return max_values


@pytest.fixture
def lower_bound() -> np.ndarray:
    """This is used to specify the min_vals for a SEPT that is either binary or randomly generated b/w 0-1"""
    min_values = np.zeros(dims, dtype=int)
    return min_values


@pytest.fixture
def reference_data() -> np.ndarray:
    """This generates random data to test the equality operators"""
    reference_data = np.random.randint(
        low=-highest, high=highest, size=(dims, dims), dtype=np.int32
    )
    return reference_data


@pytest.fixture
def reference_binary_data() -> np.ndarray:
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
            low=-highest, high=highest, size=(dims + 10, dims + 10), dtype=np.int32
        ),
        entity=ishan,
        max_vals=upper_bound,
        min_vals=lower_bound,
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


def test_add_wrong_types(
    reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
) -> None:
    """Ensure that addition with incorrect types aren't supported"""
    reference_tensor = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )
    with pytest.raises(NotImplementedError):
        _ = reference_tensor + "some string"

    with pytest.raises(NotImplementedError):
        _ = reference_tensor + dict()
        # TODO: Double check how tuples behave during addition/subtraction with np.ndarrays


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
            low=-highest, high=highest, size=(dims, dims), dtype=np.int32
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


@pytest.mark.skip(reason="GammaTensors have now been implemented")
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


def test_add_to_gamma_tensor(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    reference_scalar_manager: VirtualMachinePrivateScalarManager,
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
        entity=supreme_leader,
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
        entity=supreme_leader,
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
