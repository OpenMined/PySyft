# stdlib
from random import sample
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
from syft.core.tensor.broadcastable import is_broadcastable
from syft.core.tensor.tensor import Tensor


@pytest.fixture
def ishan() -> Entity:
    return Entity(name="Ishan")


@pytest.fixture
def traskmaster() -> Entity:
    return Entity(name="Andrew")


@pytest.fixture
def kritika() -> Entity:
    return Entity(name="Kritika")


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
def row_count() -> int:
    return np.random.randint(7) + 1


@pytest.fixture
def highest() -> int:
    return 100


@pytest.fixture
def scalar_manager() -> ScalarManager:
    return ScalarManager()


@pytest.fixture
def reference_data(highest: int, dims: int) -> np.ndarray:
    """This generates random data to test the equality operators"""
    reference_data = np.random.randint(
        low=-highest, high=highest, size=(dims, dims), dtype=np.int32
    )
    return reference_data


@pytest.fixture
def upper_bound(reference_data: np.ndarray) -> np.ndarray:
    """This is used to specify the max_vals for a SEPT that is either binary or randomly generated b/w 0-1"""
    max_values = np.ones_like(reference_data)
    return max_values


@pytest.fixture
def lower_bound(reference_data: np.ndarray) -> np.ndarray:
    """This is used to specify the min_vals for a SEPT that is either binary or randomly generated b/w 0-1"""
    min_values = np.zeros_like(reference_data)
    return min_values


@pytest.fixture
def row_data_ishan(
    highest: int,
    dims: int,
    ishan: Entity,
    scalar_manager: ScalarManager,
    row_count: int,
) -> List:
    """This generates a random number of SEPTs to populate the REPTs."""
    reference_data = []
    for _ in range(row_count):
        new_data = np.random.randint(
            low=-highest, high=highest, size=(dims, dims), dtype=np.int32
        )
        reference_data.append(
            SEPT(
                child=new_data,
                entity=ishan,
                min_vals=np.ones_like(new_data) * -highest,
                max_vals=np.ones_like(new_data) * highest,
                scalar_manager=scalar_manager,
            )
        )
    return reference_data


@pytest.fixture
def row_data_trask(
    row_count: int,
    dims: int,
    highest: int,
    traskmaster: Entity,
    scalar_manager: ScalarManager,
) -> List:
    """This generates a random number of SEPTs to populate the REPTs."""
    reference_data = []
    for _ in range(row_count):
        new_data = np.random.randint(
            low=-highest, high=highest, size=(dims, dims), dtype=np.int32
        )
        reference_data.append(
            SEPT(
                child=new_data,
                entity=traskmaster,
                min_vals=np.ones_like(new_data) * -highest,
                max_vals=np.ones_like(new_data) * highest,
                scalar_manager=scalar_manager,
            )
        )
    return reference_data


@pytest.fixture
def row_data_kritika(row_data_trask: list, kritika: Entity) -> List:
    """This generates a random number of SEPTs to populate the REPTs."""
    output = []
    for tensor in row_data_trask:
        output.append(SEPT.sept_like(tensor, kritika))
    return output


@pytest.fixture
def row_data(
    lower_bound: np.ndarray, upper_bound: np.ndarray, row_count: int, ishan: Entity
) -> List:
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
def reference_binary_data(dims: int) -> np.ndarray:
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
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
    traskmaster: Entity,
) -> None:
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
    tensor1 = REPT(rows=[data1, data2])
    tensor2 = REPT(rows=[data2, data1])
    output = tensor2 == tensor1
    assert isinstance(output, IGT)
    assert output._entities().shape == output.shape
    assert (output._values() == np.ones_like(reference_data)).all()


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


def test_add_simple_types(row_data_ishan: List, dims: int) -> None:
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
    ishan: Entity,
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


@pytest.mark.skip(reason="Need to update due to IGT changes")
def test_add_result_gamma(row_data_ishan: List, row_data_trask: List) -> None:
    """Test to see if GammaTensors are produced by adding Tensors of different entities"""
    tensor1 = REPT(rows=row_data_ishan)
    tensor2 = REPT(rows=row_data_trask)
    result = tensor2 + tensor1

    assert isinstance(result, REPT), "REPT + REPT != REPT"
    for tensor in result.child:
        print(type(tensor))
        assert isinstance(
            tensor, IGT
        ), "SEPT(entity1) + SEPT(entity2) != IGT(entity1, entity2)"


@pytest.mark.skip(reason="Need to update due to IGT changes")
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
    """Test to see if Flatten works for the ideal case"""
    reference_tensor = REPT(rows=row_data_ishan)
    output = reference_tensor.flatten()
    for sept in output:
        assert len(sept.child.shape) == 1, "Flatten shape incorrect"

    correct_output = []
    for row in row_data_ishan:
        correct_output.append(row.flatten())
    assert correct_output == output.child, "Flatten did not work as expected"


def test_ravel(row_data_ishan: List) -> None:
    """Test to see if Ravel works for the ideal case"""
    reference_tensor = REPT(rows=row_data_ishan)
    output = reference_tensor.ravel()
    for sept in output:
        assert len(sept.child.shape) == 1, "Ravel shape incorrect"

    correct_output = []
    for row in row_data_ishan:
        correct_output.append(row.ravel())
    assert correct_output == output.child, "Ravel did not work as expected"


def test_transpose(row_data_ishan: List) -> None:
    """Test to see if Transpose works for the ideal case"""
    reference_tensor = REPT(rows=row_data_ishan)
    output = reference_tensor.transpose()
    for index, sept in enumerate(output):
        assert (
            tuple(sept.shape) == reference_tensor.child[index].shape
        ), "Transpose shape incorrect"

    correct_output = []
    for row in row_data_ishan:
        correct_output.append(row.transpose())
    assert correct_output == output.child, "Transpose did not work as expected"


@pytest.mark.skip(reason="Not supporting partition for this release")
def test_partition(ishan: Entity, highest: int, dims: int) -> None:
    """Test to see if Partition works for the ideal case"""
    data = np.random.randint(
        low=-highest, high=highest, size=(dims, dims), dtype=np.int32
    )
    sept = SEPT(
        child=data,
        entity=ishan,
        min_vals=np.ones_like(data) * -100,
        max_vals=np.ones_like(data) * 100,
    )
    reference_tensor = REPT(rows=sept)

    reference_tensor.partition(kth=1)
    sept.partition(kth=1)
    assert reference_tensor.child == sept, "Partition did not work as expected"


@pytest.mark.skipif(
    dims == 1, reason="Not enough dimensions to do the compress operation"
)
def test_compress(row_data_ishan: List, ishan: Entity) -> None:
    """Test to see if Compress works for the ideal case"""
    reference_tensor = REPT(rows=row_data_ishan)

    output = reference_tensor.compress([0, 1])

    target_output = list()
    for row in row_data_ishan:
        assert row.entity == ishan
        new_row = row.compress([0, 1])
        assert new_row.entity == ishan
        target_output.append(new_row)

    for result, target in zip(output, target_output):
        assert isinstance(result.child, SEPT)
        assert isinstance(target, SEPT)
        assert result.child == target, "Compress operation failed"


def test_resize(row_data_ishan: List) -> None:
    """Test to see if Resize works for the ideal case"""
    reference_tensor = REPT(rows=row_data_ishan)
    original_tensor = reference_tensor.copy()

    new_shape = original_tensor.flatten().shape
    reference_tensor.resize(new_shape)
    assert reference_tensor.shape != original_tensor.shape, "Resize didn't work"
    assert reference_tensor.shape == new_shape, "Resize shape doesn't check out"


@pytest.mark.skipif(dims == 1, reason="Dims too low for this operation")
def test_reshape(row_data_ishan: List) -> None:
    """Test to see if Reshape works for the ideal case"""
    reference_tensor = REPT(rows=row_data_ishan)
    original_shape = reference_tensor.shape
    new_shape = tuple(
        [len(reference_tensor.child)] + [np.prod(reference_tensor.child[0].shape)] + [1]
    )
    assert new_shape[0] == reference_tensor.shape[0], "Shape isn't usable for reshape"
    output = reference_tensor.reshape(new_shape)
    assert output.shape != original_shape, "Reshape didn't change shape at all"
    assert output.shape == new_shape, "Reshape didn't change shape properly"
    assert output.shape != reference_tensor.shape, "Reshape didn't modify in-place"
    assert original_shape == reference_tensor.shape, "Reshape didn't modify in-place"


def test_squeeze(row_data_ishan: List, ishan: Entity) -> None:
    """Test to see if Squeeze works for the ideal case"""
    data = np.random.randint(low=-100, high=100, size=(10, 1, 10), dtype=np.int32)
    sept = SEPT(
        child=data,
        entity=ishan,
        min_vals=np.ones_like(data) * -100,
        max_vals=np.ones_like(data) * 100,
    )
    reference_tensor = REPT(rows=[sept])

    output = reference_tensor.squeeze()
    target = sept.squeeze()
    assert output.child[0] == target, "Squeeze did not work as expected"


def test_swapaxes(row_data_ishan: List, ishan: Entity, highest: int, dims: int) -> None:
    """Test to see if Swapaxes works for the ideal case"""
    data = np.random.randint(
        low=-highest, high=highest, size=(dims, dims), dtype=np.int32
    )
    sept = SEPT(
        child=data,
        entity=ishan,
        min_vals=np.ones_like(data) * -highest,
        max_vals=np.ones_like(data) * highest,
    )
    reference_tensor = REPT(rows=[sept])

    output = reference_tensor.swapaxes(1, 2)
    target = sept.swapaxes(0, 1)
    assert output.child[0] == target, "Swapaxes did not work as expected"


def test_mul_simple(row_data_ishan: List) -> None:
    """Ensure multiplication works with REPTs & Simple types (int/float/bool/np.ndarray)"""

    reference_tensor = REPT(rows=row_data_ishan)
    output = reference_tensor * 5
    assert (output.max_vals == 5 * reference_tensor.max_vals).all()
    assert (output.min_vals == 5 * reference_tensor.min_vals).all()
    assert output.shape == reference_tensor.shape
    assert output.child == [i * 5 for i in reference_tensor.child]


@pytest.mark.skip(reason="IGT Equality not implemented yet")
def test_mul_rept(row_data_ishan: List, row_data_trask: List) -> None:
    """Test multiplication of two REPTs"""

    # Common data
    reference_tensor1 = REPT(rows=row_data_ishan)
    reference_tensor2 = REPT(rows=row_data_trask)

    # Private-Public
    output = reference_tensor1 * reference_tensor1
    assert (
        output.max_vals == reference_tensor1.max_vals * reference_tensor1.max_vals
    ).all()
    # assert (output.min_vals == reference_tensor1.min_vals * reference_tensor1.min_vals).all()
    assert isinstance(output, REPT)
    for output_tensor, input_tensor in zip(output.child, row_data_ishan):
        assert isinstance(output_tensor, SEPT)
        assert output_tensor == input_tensor * input_tensor

    # Private-Private
    assert len(row_data_ishan) == len(row_data_trask)
    assert reference_tensor1.shape == reference_tensor2.shape
    output = reference_tensor1 * reference_tensor2
    assert isinstance(output, REPT)
    for output_tensor, ishan_tensor, trask_tensor in zip(
        output.child, row_data_ishan, row_data_trask
    ):
        assert isinstance(output_tensor, IGT)
        result = ishan_tensor * trask_tensor
        assert isinstance(result, IGT)
        print(output_tensor._values())
        print(result._values())
        assert output_tensor == result
        # assert output_tensor == ishan_tensor * trask_tensor


def test_mul_sept(
    row_data_ishan: List,
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    """Test REPT * SEPT"""
    sept = SEPT(
        child=reference_data, max_vals=upper_bound, min_vals=lower_bound, entity=ishan
    )
    rept = REPT(rows=row_data_ishan)
    if not is_broadcastable(sept.shape, rept.shape[1:]):
        print(sept.shape, rept.shape)
        with pytest.raises(Exception):
            rept * sept
    else:
        print(sept.shape, rept.shape)
        output = rept * sept
        assert isinstance(output, REPT)


def test_neg(row_data_ishan: List) -> None:
    """Test __neg__"""
    reference_tensor = REPT(rows=row_data_ishan)
    negative_tensor = reference_tensor.__neg__()

    assert reference_tensor.shape == negative_tensor.shape
    for original, negative in zip(reference_tensor, negative_tensor):
        assert negative == -original
        assert negative.child == original.child * -1
        assert (negative.min_vals == original.max_vals * -1).all()
        assert (negative.max_vals == original.min_vals * -1).all()


@pytest.mark.skip(
    reason="Test passes, but raises a Deprecation Warning for elementwise comparisons"
)
def test_and(row_count: int, ishan: Entity, dims: int) -> None:
    new_list = list()
    for _ in range(row_count):
        data = np.random.randint(2, size=(dims, dims))
        new_list.append(
            SEPT(
                child=data,
                min_vals=np.zeros_like(data),
                max_vals=np.ones_like(data),
                entity=ishan,
            )
        )
    reference_tensor = REPT(rows=new_list, check_shape=False)
    output = reference_tensor & False
    for index, tensor in enumerate(reference_tensor.child):
        assert (tensor & False) == output[index]


def test_floordiv_array(row_data_ishan: list) -> None:
    """Test floordiv with np.ndarrays"""
    reference_tensor = REPT(rows=row_data_ishan)
    other = np.ones_like(row_data_ishan[0].child)
    output = reference_tensor // other
    for index, tensor in enumerate(output.child):
        assert tensor == row_data_ishan[index] // other


def test_floordiv_sept(row_data_ishan: list) -> None:
    """Test floordiv with SEPT"""
    reference_tensor = REPT(rows=row_data_ishan)
    other = row_data_ishan[0]
    try:
        output = reference_tensor // other

        for index, tensor in enumerate(output.child):
            assert tensor == row_data_ishan[index] // other.child
    except ZeroDivisionError as e:
        print("ZeroDivisionError expected with random data", e)


def test_floordiv_rept(row_data_ishan: list) -> None:
    """Test floordiv with REPT"""
    reference_tensor = REPT(rows=row_data_ishan)
    other_data = [i // 2 + 1 for i in row_data_ishan]
    other = REPT(rows=other_data)
    try:
        output = reference_tensor // other

        for index, tensor in enumerate(output.child):
            assert tensor == row_data_ishan[index] // other_data[index]
    except ZeroDivisionError as e:
        print("ZeroDivisionError expected with random data", e)


@pytest.mark.skip(reason="Not supporting mod for 0.6.0 release")
def test_mod_array(row_data_ishan: list) -> None:
    """Test mod with np.ndarrays"""
    reference_tensor = REPT(rows=row_data_ishan)
    other = np.ones_like(row_data_ishan[0].child)
    output = reference_tensor % other
    for index, tensor in enumerate(output.child):
        assert tensor == row_data_ishan[index] % other


@pytest.mark.skip(reason="Not supporting mod for 0.6.0 release")
def test_mod_sept(row_data_ishan: list) -> None:
    """Test mod with SEPT"""
    reference_tensor = REPT(rows=row_data_ishan)
    other = row_data_ishan[0]
    try:
        output = reference_tensor % other

        for index, tensor in enumerate(output.child):
            assert tensor == row_data_ishan[index] % other.child
    except ZeroDivisionError as e:
        print("ZeroDivisionError expected with random data", e)


@pytest.mark.skip(reason="Not supporting mod for 0.6.0 release")
def test_mod_rept(row_data_ishan: list) -> None:
    """Test mod with REPT"""
    reference_tensor = REPT(rows=row_data_ishan)
    other_data = [i // 2 + 1 for i in row_data_ishan]
    other = REPT(rows=other_data)
    try:
        output = reference_tensor % other

        for index, tensor in enumerate(output.child):
            assert tensor == row_data_ishan[index] // other_data[index]
    except ZeroDivisionError as e:
        print("ZeroDivisionError expected with random data", e)


@pytest.mark.skip(reason="Not supporting mod for 0.6.0 release")
def test_divmod_array(row_data_ishan: list) -> None:
    """Test divmod with np.ndarrays"""
    reference_tensor = REPT(rows=row_data_ishan)
    other = np.ones_like(row_data_ishan[0].child)
    quotient, remainder = reference_tensor.__divmod__(other)
    for index, tensors in enumerate(zip(quotient.child, remainder.child)):
        assert tensors[0] == row_data_ishan[index] % other
        assert tensors[1] == row_data_ishan[index] % other


@pytest.mark.skip(reason="Not supporting mod for 0.6.0 release")
def test_divmod_sept(row_data_ishan: list) -> None:
    """Test divmod with SEPT"""
    reference_tensor = REPT(rows=row_data_ishan)
    other = row_data_ishan[0]
    try:
        quotient, remainder = reference_tensor.__divmod__(other)

        for index, tensors in enumerate(zip(quotient.child, remainder.child)):
            assert tensors[0] == row_data_ishan[index] // other.child
            assert tensors[1] == row_data_ishan[index] % other.child
    except ZeroDivisionError as e:
        print("ZeroDivisionError expected with random data", e)


@pytest.mark.skip(reason="Not supporting mod for 0.6.0 release")
def test_divmod_rept(row_data_ishan: list) -> None:
    """Test divmod with REPT"""
    reference_tensor = REPT(rows=row_data_ishan)
    other_data = [i // 2 + 1 for i in row_data_ishan]
    other = REPT(rows=other_data)
    try:
        quotient, remainder = reference_tensor.__divmod__(other)

        for index, tensors in enumerate(zip(quotient.child, remainder.child)):
            assert tensors[0] == row_data_ishan[index] // other_data[index]
            assert tensors[1] == row_data_ishan[index] % other_data[index]
    except ZeroDivisionError as e:
        print("ZeroDivisionError expected with random data", e)


@pytest.mark.skip(
    reason="Test passes, but raises a Deprecation Warning for elementwise comparisons"
)
def test_or(row_count: int, ishan: Entity, dims: int) -> None:
    # this test crashes the test worker somehow??
    new_list = list()
    for _ in range(row_count):
        data = np.random.randint(2, size=(dims, dims))
        new_list.append(
            SEPT(
                child=data,
                min_vals=np.zeros_like(data),
                max_vals=np.ones_like(data),
                entity=ishan,
            )
        )
    reference_tensor = REPT(rows=new_list, check_shape=False)
    output = reference_tensor | False
    for index, tensor in enumerate(reference_tensor.child):
        assert (tensor | False) == output[index]


def test_matmul_array(row_data_ishan: list) -> None:
    reference_tensor = REPT(rows=row_data_ishan)
    other = np.ones_like(row_data_ishan[0].child.T) * 5
    output = reference_tensor.__matmul__(other)

    for input_tensor, output_tensor in zip(reference_tensor.child, output.child):
        assert output_tensor.shape[1] == reference_tensor.shape[1]
        assert output_tensor.shape[-1] == other.shape[-1]
        assert output_tensor == input_tensor.__matmul__(other)


def test_matmul_sept(
    row_data_ishan: list,
    reference_data: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    ishan: Entity,
) -> None:
    reference_tensor = REPT(rows=row_data_ishan)
    other = row_data_ishan[0].transpose() * 2
    output = reference_tensor.__matmul__(other)

    for input_tensor, output_tensor in zip(reference_tensor.child, output.child):
        assert output_tensor.shape[1] == reference_tensor.shape[1]
        assert output_tensor.shape[-1] == other.shape[-1]
        assert output_tensor == input_tensor.__matmul__(other.child)


def test_matmul_rept(row_data_ishan: list) -> None:
    reference_tensor = REPT(rows=row_data_ishan)
    data = [i.transpose() * 2 for i in row_data_ishan]
    other = REPT(rows=data)
    output = reference_tensor.__matmul__(other)

    for input_tensor, other_tensor, output_tensor in zip(
        reference_tensor.child, other.child, output.child
    ):
        assert output_tensor.shape[1] == reference_tensor.shape[1]
        assert output_tensor.shape[-1] == other.shape[-1]
        assert output_tensor == input_tensor.child.__matmul__(other_tensor.child)


def test_cumsum(row_data_ishan: list) -> None:
    reference_tensor = REPT(rows=row_data_ishan)
    output = reference_tensor.cumsum()
    for index, row in enumerate(row_data_ishan):
        assert (output.child[index] == row.cumsum()).child.all()


def test_cumprod(row_data_ishan: list) -> None:
    reference_tensor = REPT(rows=row_data_ishan)
    output = reference_tensor.cumprod()
    for index, row in enumerate(row_data_ishan):
        assert (output.child[index] == row.cumprod()).child.all()


def test_round(row_data_ishan: List) -> None:
    reference_tensor = REPT(rows=row_data_ishan)
    for target, output in zip(row_data_ishan, reference_tensor.round(decimals=0).child):
        assert (target.child.astype(np.int32) == output.child).all()


def test_entities(row_data_ishan: list, ishan: Entity, traskmaster: Entity) -> None:
    """Test that n_entities works as intended"""
    rept_tensor1 = REPT(rows=row_data_ishan)
    assert rept_tensor1.n_entities == 1

    rept_tensor2 = REPT(
        rows=[
            SEPT(
                child=np.random.randint(low=0, high=20, size=(10, 10)),
                min_vals=np.zeros((10, 10), dtype=np.int32) * 20,
                max_vals=np.ones((10, 10), dtype=np.int32),
                entity=ishan,
            ),
            SEPT(
                child=np.random.randint(low=0, high=60, size=(10, 10)),
                min_vals=np.zeros((10, 10), dtype=np.int32) * 60,
                max_vals=np.ones((10, 10), dtype=np.int32),
                entity=traskmaster,
            ),
        ]
    )
    assert rept_tensor2.n_entities == 2

    rept_tensor3 = REPT(
        rows=[
            SEPT(
                child=np.random.randint(low=0, high=20, size=(10, 10)),
                min_vals=np.zeros((10, 10), dtype=np.int32) * 20,
                max_vals=np.ones((10, 10), dtype=np.int32),
                entity=ishan,
            ),
            SEPT(
                child=np.random.randint(low=0, high=60, size=(10, 10)),
                min_vals=np.zeros((10, 10), dtype=np.int32) * 60,
                max_vals=np.ones((10, 10), dtype=np.int32),
                entity=traskmaster,
            ),
            SEPT(
                child=np.random.randint(low=0, high=40, size=(10, 10)),
                min_vals=np.zeros((10, 10), dtype=np.int32) * 40,
                max_vals=np.ones((10, 10), dtype=np.int32),
                entity=ishan,
            ),
        ]
    )
    assert rept_tensor3.n_entities == 2


@pytest.fixture
def pos_row_data(
    row_count: int,
    dims: int,
    highest: int,
    traskmaster: Entity,
    scalar_manager: ScalarManager,
) -> List:
    """This generates a random number of SEPTs to populate the REPTs."""
    reference_data = []
    for _ in range(row_count):
        new_data = np.random.randint(
            low=1, high=highest, size=(dims, dims), dtype=np.int32
        )
        reference_data.append(
            SEPT(
                child=new_data,
                entity=traskmaster,
                min_vals=np.ones_like(new_data) * -highest,
                max_vals=np.ones_like(new_data) * highest,
                scalar_manager=scalar_manager,
            )
        )
    return reference_data


def test_le_same_entities(row_data_trask: List) -> None:
    tensor = REPT(rows=row_data_trask)
    second_tensor = REPT(rows=row_data_trask)
    third_tensor = REPT(rows=[x + 1 for x in row_data_trask])
    assert tensor.shape == second_tensor.shape
    assert tensor.shape == third_tensor.shape
    output1 = tensor <= second_tensor
    for i in range(len(tensor.child)):
        assert (output1).child[i].child.all()
    output2 = tensor <= third_tensor
    for i in range(len(tensor.child)):
        assert (output2).child[i].child.all()


def test_le_diff_entities(row_data_trask: List, row_data_kritika: List) -> None:
    tensor = REPT(rows=row_data_trask)
    second_tensor = REPT(rows=row_data_kritika)
    assert tensor.shape == second_tensor.shape
    output = tensor <= second_tensor
    assert isinstance(output, IGT)
    assert (output._values() == np.ones_like(output._values())).all()


def test_ge_same_entities(row_data_trask: List) -> None:
    tensor = REPT(rows=row_data_trask)
    second_tensor = REPT(rows=row_data_trask)
    third_tensor = REPT(rows=[x + 1 for x in row_data_trask])
    assert tensor.shape == second_tensor.shape
    assert tensor.shape == third_tensor.shape
    output1 = tensor >= second_tensor
    for i in range(len(tensor.child)):
        assert (output1).child[i].child.all()
    output2 = third_tensor >= tensor
    for i in range(len(tensor.child)):
        assert (output2).child[i].child.all()


def test_ge_diff_entities(row_data_trask: List, row_data_kritika: List) -> None:
    tensor = REPT(rows=row_data_trask)
    second_tensor = REPT(rows=row_data_kritika)
    assert tensor.shape == second_tensor.shape
    output = tensor >= second_tensor
    assert isinstance(output, IGT)
    assert (output._values() == np.ones_like(output._values())).all()


def test_lt_same_entities(row_data_trask: List) -> None:
    tensor = REPT(rows=row_data_trask)
    second_tensor = REPT(rows=row_data_trask)
    third_tensor = REPT(rows=[x + 1 for x in row_data_trask])
    assert tensor.shape == second_tensor.shape
    assert tensor.shape == third_tensor.shape
    output1 = tensor < second_tensor
    for i in range(len(tensor.child)):
        assert not (output1).child[i].child.all()
    output2 = tensor < third_tensor
    for i in range(len(tensor.child)):
        assert (output2).child[i].child.all()


def test_lt_diff_entities(row_data_trask: List, row_data_kritika: List) -> None:
    tensor = REPT(rows=row_data_trask)
    second_tensor = REPT(rows=row_data_kritika)
    assert tensor.shape == second_tensor.shape
    output = tensor < second_tensor
    assert isinstance(output, IGT)
    assert (output._values() == np.zeros_like(output._values())).all()


def test_gt_same_entities(row_data_trask: List) -> None:
    tensor = REPT(rows=row_data_trask)
    second_tensor = REPT(rows=row_data_trask)
    third_tensor = REPT(rows=[x + 1 for x in row_data_trask])
    assert tensor.shape == second_tensor.shape
    assert tensor.shape == third_tensor.shape
    output1 = tensor > second_tensor
    for i in range(len(tensor.child)):
        assert not (output1).child[i].child.all()
    output2 = third_tensor > tensor
    for i in range(len(tensor.child)):
        assert (output2).child[i].child.all()


def test_gt_diff_entities(row_data_trask: List, row_data_kritika: List) -> None:
    tensor = REPT(rows=row_data_trask)
    second_tensor = REPT(rows=row_data_kritika)
    assert tensor.shape == second_tensor.shape
    output = tensor > second_tensor
    assert isinstance(output, IGT)
    assert (output._values() == np.zeros_like(output._values())).all()


def test_clip(row_data_trask: List, highest: int) -> None:
    clip_min = np.random.randint(-highest, highest / 2)
    clip_max = np.random.randint(highest / 2, highest)

    tensor = REPT(rows=row_data_trask)
    clipped_tensor1 = tensor.clip(clip_min, clip_max)
    clipped_tensor2 = tensor.clip(clip_max, clip_min)
    clipped_tensor3 = tensor.clip(clip_min, None)
    assert clipped_tensor1.shape == tensor.shape
    assert clipped_tensor2.shape == tensor.shape
    assert clipped_tensor3.shape == tensor.shape
    for i in range(len(tensor.child)):
        assert (clipped_tensor1.child[i].child >= clip_min).all() & (
            clipped_tensor1.child[i].child <= clip_max
        ).all()
    for i in range(len(tensor.child)):
        assert (clipped_tensor2.child[i].child == clip_min).all()
    for i in range(len(tensor.child)):
        assert (clipped_tensor3.child[i].child >= clip_min).all()


def test_any(pos_row_data: List) -> None:
    zeros_tensor = (
        REPT(
            rows=pos_row_data,
        )
        * 0
    )
    pos_tensor = REPT(
        rows=pos_row_data,
    )
    any_zeros_tensor = zeros_tensor.any()
    for i in range(len(zeros_tensor.child)):
        assert not any_zeros_tensor.child[i].child
    any_pos_tensor = pos_tensor.any()
    for i in range(len(pos_tensor.child)):
        assert any_pos_tensor.child[i].child


def test_all(pos_row_data: List) -> None:
    zeros_tensor = (
        REPT(
            rows=pos_row_data,
        )
        * 0
    )
    pos_tensor = REPT(
        rows=pos_row_data,
    )
    all_zeros_tensor = zeros_tensor.all()
    for i in range(len(zeros_tensor.child)):
        assert not all_zeros_tensor.child[i].child
    all_pos_tensor = pos_tensor.all()
    for i in range(len(pos_tensor.child)):
        assert all_pos_tensor.child[i].child


def test_abs(pos_row_data: List) -> None:
    tensor = REPT(rows=pos_row_data)
    neg_tensor = REPT(rows=pos_row_data) * -1
    abs_neg_tensor = neg_tensor.abs()
    for i in range(len(tensor.child)):
        assert (abs_neg_tensor.child[i].child == tensor.child[i].child).all()


def test_pow(row_data_trask: List) -> None:
    rand_pow = np.random.randint(1, 10)
    tensor = REPT(rows=row_data_trask)
    pow_tensor = tensor.pow(rand_pow)
    assert pow_tensor.shape == tensor.shape
    for i in range(len(tensor.child)):
        assert (pow_tensor.child[i].child == tensor.child[i].child ** rand_pow).all()


def test_sum(
    row_data_trask: List,
    dims: int,
) -> None:
    tensor = REPT(rows=row_data_trask)
    sum_tensor = tensor.sum()
    for k in range(len(tensor.child)):
        tensor_sum = 0
        for i in range(dims):
            for j in range(dims):
                tensor_sum += tensor.child[k].child[i, j]
        assert sum_tensor.child[k].child == tensor_sum


def test_copy(row_data_trask: List) -> None:
    tensor = REPT(rows=row_data_trask)
    tensor_copy = tensor.copy()
    for i in range(len(tensor.child)):
        assert (tensor_copy.child[i].child == tensor.child[i].child).all()


def test_take(row_data_trask: List, dims: int) -> None:
    tensor = REPT(rows=row_data_trask)
    indices = sample(range(dims), dims)
    tensor_take = tensor.take(indices)
    for i in range(len(tensor)):
        assert (tensor_take.child[i].child == tensor.child[i].child[0][indices]).all()


def test_diagonal(row_data_trask: List, dims: int) -> None:
    tensor = REPT(rows=row_data_trask)
    tensor_diagonal = tensor.diagonal()
    for i in range(len(tensor)):
        for j in range(dims):
            assert (
                tensor_diagonal.child[i].child[j] == tensor.child[i].child[j][j]
            ).all()


def test_converter(
    row_data_ishan: List,
    traskmaster: Entity,
    ishan: Entity,
    highest: int,
    scalar_manager: ScalarManager,
) -> None:
    # Test that SEPTs can be converted
    output = REPT.convert_to_gamma(row_data_ishan)
    assert isinstance(output, IGT)
    assert output._entities().shape == output.shape

    new_data = row_data_ishan[0].child

    # Test with just a list of IGTs
    igt1 = SEPT(
        child=new_data,
        entity=traskmaster,
        min_vals=np.ones_like(new_data) * -highest,
        max_vals=np.ones_like(new_data) * highest,
        scalar_manager=scalar_manager,
    ) + SEPT(
        child=new_data,
        entity=ishan,
        min_vals=np.ones_like(new_data) * -highest,
        max_vals=np.ones_like(new_data) * highest,
        scalar_manager=scalar_manager,
    )
    igt2 = igt1 + 1
    assert isinstance(igt1, IGT)
    output = REPT.convert_to_gamma([igt1, igt2])
    assert isinstance(output, IGT)

    # Test hybrid
    assert new_data.shape == igt1.shape
    output = REPT.convert_to_gamma([igt1, row_data_ishan[0]])
    assert isinstance(output, IGT)
    assert output._entities().shape == output.shape
