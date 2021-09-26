# stdlib
from random import randint
from typing import List

# third party
import numpy as np
import pytest

# syft absolute
from syft.core.adp.entity import Entity
from syft.core.adp.vm_private_scalar_manager import (
    VirtualMachinePrivateScalarManager as ScalarManager,
)
from syft.core.tensor.autodp.intermediate_gamma import IntermediateGammaTensor as IGT
from syft.core.tensor.autodp.row_entity_phi import RowEntityPhiTensor as REPT
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor as SEPT
from syft.core.tensor.tensor import Tensor

# ------------------- EQUALITY OPERATORS -----------------------------------------------

# Global constants
ishan = Entity(name="Ishan")
traskmaster = Entity(name="Trask")
dims = max(3, np.random.randint(10) + 3)  # Avoid size 0, 1
row_count = max(3, np.random.randint(7) + 1)  # Avoids size 0, 1
scalar_manager = ScalarManager()


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
def row_data_ishan() -> List:
    """This generates a random number of SEPTs to populate the REPTs."""
    reference_data = []
    for _ in range(row_count):
        new_data = np.random.random((dims, dims))
        reference_data.append(
            SEPT(
                child=new_data,
                entity=ishan,
                min_vals=np.zeros_like(new_data),
                max_vals=np.ones_like(new_data),
                scalar_manager=scalar_manager,
            )
        )
    return reference_data


@pytest.fixture
def row_data_trask() -> List:
    """This generates a random number of SEPTs to populate the REPTs."""
    reference_data = []
    for _ in range(row_count):
        new_data = np.random.random((dims, dims))
        reference_data.append(
            SEPT(
                child=new_data,
                entity=traskmaster,
                min_vals=np.zeros_like(new_data),
                max_vals=np.ones_like(new_data),
                scalar_manager=scalar_manager,
            )
        )
    return reference_data


@pytest.fixture
def row_data(lower_bound: np.ndarray, upper_bound: np.ndarray) -> List:
    """This generates a random number of SEPTs to populate the REPTs."""
    reference_data = []
    for _ in range(row_count):
        reference_data.append(
            SEPT(
                child=np.random.random((dims, dims)),
                entity=ishan,
                min_vals=lower_bound,
                max_vals=upper_bound,
            )
        )
    return reference_data


@pytest.fixture
def reference_binary_data() -> np.ndarray:
    """Generate binary data to test the equality operators with bools"""
    binary_data = np.random.randint(2, size=(dims, dims))
    return binary_data


def test_eq(row_data_ishan: List) -> None:
    """Test equality between two identical RowEntityPhiTensors"""
    reference_tensor = REPT(rows=row_data_ishan)
    second_tensor = REPT(rows=row_data_ishan)
    third_tensor = reference_tensor

    assert reference_tensor == second_tensor, "Identical Tensors don't match up"
    assert reference_tensor == third_tensor, "Identical Tensors don't match up"


def test_eq_diff_tensors(row_data_ishan: List) -> None:
    """Here we're testing equality between a REPT and other tensor types."""

    # Narrow row data down to a single data point (SEPT)
    sept_data: SEPT = row_data_ishan[0]
    reference_tensor = REPT(rows=sept_data)
    reference_sept = sept_data

    assert (
        reference_tensor == reference_sept
    ), "REPT and SEPT equality comparison failed"
    assert sept_data == reference_tensor.child, "Error: data & child don't match"
    assert (
        type(reference_tensor == reference_sept) == REPT
    ), "Return type error for equality comparison b/w REPT, SEPT"
    # assert type(reference_sept == reference_tensor) == REPT, "Return type error for == comparison b/w SEPT, REPT"


def test_eq_diff_entities(
    row_data_ishan: List,
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
) -> REPT:
    """Test equality between REPTs with different owners"""
    data1 = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )
    data2 = SEPT(
        child=reference_data,
        max_vals=upper_bound,
        min_vals=lower_bound,
        entity=traskmaster,
    )
    tensor1 = REPT(rows=data1)
    tensor2 = REPT(rows=data2)

    with pytest.raises(NotImplementedError):
        return tensor1 == tensor2


# TODO: Update this test after REPT.all() and .any() are implemented, and check `assert not comparison_result`
def test_eq_values(
    row_data_ishan: List,
    reference_data: np.ndarray,
) -> None:
    """Test REPTs belonging to the same owner, with different data"""
    tensor1 = REPT(rows=row_data_ishan)
    tensor2 = REPT(rows=row_data_ishan) + 1

    assert tensor2.shape == tensor1.shape, "Tensors not initialized properly"
    # assert tensor2 != tensor1, "Error: REPT + 1 == REPT"  # TODO: Investigate RecursionError Here

    # Debug test issues
    assert isinstance(tensor2.child[0], SEPT)
    assert isinstance(tensor1.child[0], SEPT)
    assert tensor2.child[0] != tensor1.child[0]
    assert isinstance(
        tensor2.child[0] != tensor1.child[0], SEPT
    ), "Underlying SEPT comparison yields wrong type"
    assert isinstance(tensor2 == tensor1, REPT), "REPT == REPT, Output has wrong type"
    assert not (tensor2 == tensor1).child[0].child.any()
    for i in range(len(tensor2.child)):
        # Explicitly checks that comparison_result below is correct
        assert (
            not (tensor2 == tensor1).child[i].child.any()
        ), f"REPT + 1 == REPT failed at child {i}"

    # comparison_result = tensor1 == tensor2
    tensor1 == tensor2
    # assert not comparison_result  # This will work as soon as the .all() or .any() methods are implemented.
    # Would this be more user-friendly if SEPT == SEPT -> singular T/F instead of array of T/F?


def test_ne_shapes(
    row_data_ishan: List,
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
) -> None:
    """Test REPTs belonging to the same owner, with different shapes"""
    tensor1 = REPT(rows=row_data_ishan)
    tensor2 = REPT(rows=row_data_ishan + row_data_ishan)
    assert (
        tensor2.shape != tensor1.shape
    ), "Tensors not initialized properly for this test"

    with pytest.raises(Exception):
        _ = tensor2 == tensor1


@pytest.mark.skip(
    reason="Comparison works but throws Depreciation Warning preventing merge"
)
def test_eq_ndarray(row_data_ishan: List) -> None:
    """Test equality between a SEPT and a simple type (int, float, bool, np.ndarray)"""
    sub_row_data_ishan: SEPT = row_data_ishan[0]

    reference_tensor = REPT(rows=sub_row_data_ishan)
    assert sub_row_data_ishan.child == reference_tensor, "Comparison b/w REPT and "


@pytest.mark.skip(
    reason="REPT addition currently doesn't catch incorrect types (str, dict, etc)"
)
def test_add_wrong_types(row_data_ishan: List) -> None:
    """Ensure that addition with incorrect types aren't supported"""
    reference_tensor = REPT(rows=row_data_ishan)
    with pytest.raises(NotImplementedError):
        _ = reference_tensor + "some string"

    with pytest.raises(NotImplementedError):
        _ = reference_tensor + dict()
        # TODO: Double check how tuples behave during addition/subtraction with np.ndarrays


def test_add_simple_types(row_data_ishan: List) -> None:
    """Test addition of a REPT with simple types (float, ints, bools, etc)"""
    tensor = REPT(rows=row_data_ishan)
    random_int = np.random.randint(low=15, high=1000)
    result = tensor + random_int
    assert isinstance(result, REPT), "REPT + int != REPT"
    assert result[0] == tensor[0] + random_int, "Addition did not work as intended"
    assert result[-1] == tensor[-1] + random_int, "Addition did not work as intended"

    random_float = random_int * np.random.rand()
    result = tensor + random_float
    assert isinstance(result, REPT), "REPT + float != REPT"
    assert result[0] == tensor[0] + random_float, "Addition did not work as intended"
    assert result[-1] == tensor[-1] + random_float, "Addition did not work as intended"

    random_ndarray = np.random.random((dims, dims))
    result = tensor + random_ndarray
    assert isinstance(result, REPT), "SEPT + np.ndarray != SEPT"


@pytest.mark.skip(reason="Temporary")
def test_add_tensor_types(row_data_ishan: List) -> None:
    """Test addition of a REPT with various other kinds of Tensors"""
    reference_tensor = REPT(rows=row_data_ishan)
    simple_tensor = Tensor(child=np.random.random((dims, dims)))
    assert len(simple_tensor.child) == len(
        reference_tensor.child
    ), "Addition can't be performed"

    with pytest.raises(NotImplementedError):
        result = reference_tensor + simple_tensor
        assert isinstance(result, REPT), "REPT + Tensor != SEPT"
        assert (
            result.max_vals == reference_tensor.max_vals + simple_tensor.child.max()
        ), "REPT + Tensor: incorrect max_val"
        assert (
            result.min_vals == reference_tensor.min_vals + simple_tensor.child.min()
        ), "REPT + Tensor: incorrect min_val"


@pytest.mark.skip(
    reason="REPT + SEPT --> GammaTensor, but this hasn't been implemented yet"
)
def test_add_single_entity(
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    row_data_ishan: List,
) -> None:
    """Test the addition of REPT + SEPT"""
    tensor1 = REPT(rows=row_data_ishan)
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


# TODO: Fix REPT.min_vals and REPT.max_vals properties
def test_add_row_entities(row_data_ishan: List) -> None:
    """Test normal addition of two REPTs"""
    tensor1 = REPT(rows=row_data_ishan)
    tensor2 = tensor1 + tensor1
    assert isinstance(tensor2, REPT), "Error: REPT + REPT != REPT "
    assert (
        tensor2.shape == tensor1.shape
    ), "Error: REPT + REPT changed shape of resultant REPT"
    assert tensor2 == tensor1 + tensor1, "Error: REPT + REPT failed"
    # assert (tensor2.min_vals == 2 * tensor1.min_vals), "REPT + REPT results in incorrect min_vals"
    # assert (tensor2.max_vals == 2 * tensor1.max_vals), "REPT + REPT results in incorrect max_vals"

    random_offset = np.random.randint(low=10, high=100) * np.random.random()
    tensor3 = tensor2 + tensor2 + random_offset
    assert isinstance(tensor3, REPT), "Error: REPT + REPT != REPT "
    assert (
        tensor3.shape == tensor2.shape
    ), "Error: REPT + REPT changed shape of resultant REPT"
    assert tensor3 - tensor2 + tensor2 == random_offset
    # assert (tensor3.min_vals == 2 * tensor2.min_vals + random_offset), "REPT + REPT results in incorrect min_vals"
    # assert (tensor3.max_vals == 2 * tensor2.max_vals + random_offset), "REPT + REPT results in incorrect max_vals"
    return None


def test_add_sub_equivalence(row_data_ishan: List) -> None:
    """Test to see if addition of -ve and subtraction of +ve produce the same results"""
    tensor1 = REPT(rows=row_data_ishan)
    tensor2 = tensor1 * 2
    assert tensor2.shape == tensor1.shape, "REPTs initialized incorrectly"

    assert tensor1 - 5 == tensor1 + 5 * -1, "Addition of -ve != Subtraction of +ve"
    assert (
        tensor2 - tensor1 == tensor2 + tensor1 * -1
    ), "Addition of -ve != Subtraction of +ve"


def test_add_result_gamma(row_data_ishan: List, row_data_trask: List) -> None:
    """Test to see if GammaTensors are produced by adding Tensors of different entities"""
    tensor1 = REPT(rows=row_data_ishan)
    tensor2 = REPT(rows=row_data_trask)
    result = tensor2 + tensor1

    assert isinstance(result, REPT), "REPT + REPT != REPT"
    for tensor in result.child:
        assert isinstance(
            tensor, IGT
        ), "SEPT(entity1) + SEPT(entity2) != IGT(entity1, entity2)"


def test_sub_result_gamma(row_data_ishan: List, row_data_trask: List) -> None:
    """Test to see if GammaTensors are produced by subtracting Tensors of different entities"""
    tensor1 = REPT(rows=row_data_ishan)
    tensor2 = REPT(rows=row_data_trask)
    result = tensor2 - tensor1

    assert isinstance(result, REPT), "REPT + REPT != REPT"
    for tensor in result.child:
        assert isinstance(
            tensor, IGT
        ), "SEPT(entity1) + SEPT(entity2) != IGT(entity1, entity2)"


def test_flatten(row_data_ishan: List) -> None:
    """ Test to see if Flatten works for the ideal case """
    reference_tensor = REPT(rows=row_data_ishan)
    output = reference_tensor.flatten()
    for sept in output:
        assert len(sept.child.shape) == 1, "Flatten shape incorrect"

    correct_output = []
    for row in row_data_ishan:
        correct_output.append(row.flatten())
    assert correct_output == output.child, "Flatten did not work as expected"


def test_ravel(row_data_ishan: List) -> None:
    """ Test to see if Ravel works for the ideal case """
    reference_tensor = REPT(rows=row_data_ishan)
    output = reference_tensor.ravel()
    for sept in output:
        assert len(sept.child.shape) == 1, "Ravel shape incorrect"

    correct_output = []
    for row in row_data_ishan:
        correct_output.append(row.ravel())
    assert correct_output == output.child, "Ravel did not work as expected"


def test_transpose(row_data_ishan: List) -> None:
    """ Test to see if Transpose works for the ideal case """
    reference_tensor = REPT(rows=row_data_ishan)
    output = reference_tensor.transpose()
    for index, sept in enumerate(output):
        assert tuple(sept.shape) == reference_tensor.child[index].shape, "Transpose shape incorrect"

    correct_output = []
    for row in row_data_ishan:
        correct_output.append(row.transpose())
    assert correct_output == output.child, "Transpose did not work as expected"


def test_partition() -> None:
    """ Test to see if Partition works for the ideal case """
    data = np.random.randint(low=-100, high=100, size=(10, 10), dtype=np.int32)
    sept = SEPT(child=data, entity=ishan, min_vals=np.ones_like(data) * -100, max_vals=np.ones_like(data) * 100)
    reference_tensor = REPT(rows=[sept])

    reference_tensor.partition(kth=1)
    sept.partition(kth=1)
    assert reference_tensor == sept, "Partition did not work as expected"


def test_compress(row_data_ishan: List) -> None:
    """ Test to see if Compress works for the ideal case """
    pass


def test_resize(row_data_ishan: List) -> None:
    """ Test to see if Resize works for the ideal case """
    reference_tensor = REPT(rows=row_data_ishan)
    original_tensor = reference_tensor.copy()

    new_shape = original_tensor.flatten().shape
    reference_tensor.resize(new_shape)
    assert reference_tensor.shape != original_tensor.shape, "Resize didn't work"
    assert reference_tensor.shape == new_shape, "Resize shape doesn't check out"


def test_reshape(row_data_ishan: List) -> None:
    """ Test to see if Reshape works for the ideal case """
    reference_tensor = REPT(rows=row_data_ishan)
    original_shape = reference_tensor.shape
    new_shape = [len(reference_tensor.child)] + [np.prod(reference_tensor.child[0].shape)] + [1]
    assert new_shape[0] == reference_tensor.shape[0], "Shape isn't usable for reshape"
    output = reference_tensor.reshape(new_shape)
    assert output.shape != original_shape, "Reshape didn't change shape at all"
    assert output.shape == new_shape, "Reshape didn't change shape properly"
    assert output.shape != reference_tensor.shape, "Reshape didn't modify in-place"
    assert original_shape == reference_tensor.shape, "Reshape didn't modify in-place"


def test_squeeze(row_data_ishan: List) -> None:
    """ Test to see if Squeeze works for the ideal case """
    pass


def test_swapaxes(row_data_ishan: List) -> None:
    """ Test to see if Swapaxes works for the ideal case """
    pass




ent = Entity(name="test")
ent2 = Entity(name="test2")

dims = np.random.randint(10) + 1
row_count = np.random.randint(10) + 1


def rept(low, high, entity) -> List:
    data = []
    for _ in range(row_count):
        data.append(
            SEPT(
                child=np.random.randint(low=low, high=high, size=dims),
                entity=entity,
                max_vals=np.full(dims, high - 1, dtype=np.int32),
                min_vals=np.full(dims, low, dtype=np.int32),
            )
        )
    return REPT(rows=data, check_shape=False)


tensor1 = rept(-2, 4, ent)
# same entity, same data
tensor2 = tensor1
# same entity, different data
tensor3 = rept(4, 7, ent)

simple_type1 = randint(-6, -4)
simple_type2 = randint(4, 6)


def test_le() -> None:
    for i in tensor1.__le__(tensor2).child:
        assert i.child.all()
    for i in tensor1.__le__(tensor3).child:
        assert i.child.all()
    for i in tensor1.__le__(simple_type1).child:
        assert not i.child.all()
    for i in tensor1.__le__(simple_type2).child:
        assert i.child.all()


def test_ge() -> None:
    for i in tensor1.__ge__(tensor2).child:
        assert i.child.all()
    for i in tensor1.__ge__(tensor3).child:
        assert not i.child.all()
    for i in tensor1.__ge__(simple_type1).child:
        assert i.child.all()
    for i in tensor1.__ge__(simple_type2).child:
        assert not i.child.all()


def test_lt() -> None:
    for i in tensor1.__lt__(tensor2).child:
        assert not i.child.all()
    for i in tensor1.__lt__(tensor3).child:
        assert i.child.all()
    for i in tensor1.__lt__(simple_type1).child:
        assert not i.child.all()
    for i in tensor1.__lt__(simple_type2).child:
        assert i.child.all()


def test_gt() -> None:
    for i in tensor1.__gt__(tensor2).child:
        assert not i.child.all()
    for i in tensor1.__gt__(tensor3).child:
        assert not i.child.all()
    for i in tensor1.__gt__(simple_type1).child:
        assert i.child.all()
    for i in tensor1.__gt__(simple_type2).child:
        assert not i.child.all()


rand1 = np.random.randint(-4, 1)
rand2 = np.random.randint(1, 5)
clipped_tensor1 = tensor1.clip(rand1, rand2).child
clipped_tensor2 = tensor1.clip(rand2, rand1).child
clipped_tensor3 = tensor1.clip(rand1, None).child


def test_clip() -> None:
    for i in clipped_tensor1:
        assert ((i.child >= rand1) & (i.child <= rand2)).all()
    for i in clipped_tensor2:
        assert (i.child == rand1).all()
    for i in clipped_tensor3:
        assert (i.child >= rand1).all()
