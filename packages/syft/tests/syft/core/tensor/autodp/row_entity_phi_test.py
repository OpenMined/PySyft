# stdlib
from typing import List

# third party
import numpy as np
import pytest

# syft absolute
from syft.core.adp.entity import Entity
from syft.core.tensor.autodp.row_entity_phi import RowEntityPhiTensor as REPT
from syft.core.tensor.autodp.single_entity_phi import SingleEntityPhiTensor as SEPT
from syft.core.tensor.tensor import Tensor

# ------------------- EQUALITY OPERATORS -----------------------------------------------

# Global constants
ishan = Entity(name="Ishan")
supreme_leader = Entity(name="Trask")
dims = np.random.randint(10) + 1  # Avoid size 0
row_count = np.random.randint(7) + 1


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


def test_eq(row_data: List) -> None:
    """Test equality between two identical RowEntityPhiTensors"""
    reference_tensor = REPT(rows=row_data)
    second_tensor = REPT(rows=row_data)
    third_tensor = reference_tensor

    assert reference_tensor == second_tensor, "Identical Tensors don't match up"
    assert reference_tensor == third_tensor, "Identical Tensors don't match up"


def test_eq_diff_tensors(row_data: List) -> None:
    """Here we're testing equality between a REPT and other tensor types."""

    # Narrow row data down to a single data point (SEPT)
    row_data: SEPT = row_data[0]
    reference_tensor = REPT(rows=row_data)
    reference_sept = row_data

    assert (
        reference_tensor == reference_sept
    ), "REPT and SEPT equality comparison failed"
    assert row_data == reference_tensor.child, "Error: data & child don't match"
    assert (
        type(reference_tensor == reference_sept) == REPT
    ), "Return type error for equality comparison b/w REPT, SEPT"
    # assert type(reference_sept == reference_tensor) == REPT, "Return type error for == comparison b/w SEPT, REPT"


def test_eq_diff_entities(
    row_data: List,
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
        entity=supreme_leader,
    )
    tensor1 = REPT(rows=data1)
    tensor2 = REPT(rows=data2)

    with pytest.raises(NotImplementedError):
        return tensor1 == tensor2


def test_eq_ndarray(row_data: List) -> None:
    """Test equality between a SEPT and a simple type (int, float, bool, np.ndarray)"""
    sub_row_data: SEPT = row_data[0]

    reference_tensor = REPT(rows=sub_row_data)
    assert sub_row_data.child == reference_tensor, "Comparison b/w REPT and "


@pytest.mark.skip(reason="REPT addition currently doesn't catch incorrect types (str, dict, etc)")
def test_add_wrong_types(
        row_data: List
) -> None:
    """ Ensure that addition with incorrect types aren't supported"""
    reference_tensor = REPT(rows=row_data)
    with pytest.raises(NotImplementedError):
        result1 = reference_tensor + "some string"
        result2 = reference_tensor + dict()
        # TODO: Double check how tuples behave during addition/subtraction with np.ndarrays
    return None


def test_add_simple_types(
        row_data: List
) -> None:
    """ Test addition of a REPT with simple types (float, ints, bools, etc) """
    tensor = REPT(rows=row_data)

    random_int = np.random.randint(low=15, high=1000)
    result = tensor + random_int
    assert isinstance(result, REPT), "REPT + int != REPT"
    assert (result[0] == tensor[0] + random_int), "Addition did not work as intended"
    assert (result[-1] == tensor[-1] + random_int), "Addition did not work as intended"

    random_float = random_int * np.random.rand()
    result = tensor + random_float
    assert isinstance(result, REPT), "REPT + float != REPT"
    assert (result[0] == tensor[0] + random_float), "Addition did not work as intended"
    assert (result[-1] == tensor[-1] + random_float), "Addition did not work as intended"

    random_ndarray = np.random.random((dims, dims))
    result = tensor + random_ndarray
    assert isinstance(result, REPT), "SEPT + np.ndarray != SEPT"

    return None


def test_add_tensor_types(row_data: List) -> None:
    """ Test addition of a REPT with various other kinds of Tensors"""

    reference_tensor = REPT(rows=row_data)
    simple_tensor = Tensor(child=np.random.random((dims, dims)))
    assert len(simple_tensor.child) == len(reference_tensor.child), "Addition can't be performed"

    with pytest.raises(NotImplementedError):
        result = reference_tensor + simple_tensor
        assert isinstance(result, REPT), "REPT + Tensor != SEPT"
        assert (result.max_vals == reference_tensor.max_vals + simple_tensor.child.max()), "REPT + Tensor: incorrect max_val"
        assert (result.min_vals == reference_tensor.min_vals + simple_tensor.child.min()), "REPT + Tensor: incorrect min_val"
        return None


@pytest.mark.skip(reason="REPT + SEPT --> GammaTensor, but this hasn't been implemented yet")
def test_add_single_entity(
        reference_data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray, row_data: List
) -> None:
    """Test the addition of REPT + SEPT"""
    tensor1 = REPT(rows=row_data)
    tensor2 = SEPT(
        child=reference_data, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    result = tensor2 + tensor1
    assert isinstance(result, SEPT), "Addition of two SEPTs is wrong type"
    assert (result.max_vals == 2 * upper_bound).all(), "Addition of two SEPTs results in incorrect max_val"
    assert (result.min_vals == 2 * lower_bound).all(), "Addition of two SEPTs results in incorrect min_val"

    # Try with negative values
    tensor3 = SEPT(
        child=reference_data * -1.5, entity=ishan, max_vals=upper_bound, min_vals=lower_bound
    )

    result = tensor3 + tensor1
    assert isinstance(result, SEPT), "Addition of two SEPTs is wrong type"
    assert (result.max_vals == tensor3.max_vals + tensor1.max_vals).all(), "SEPT + SEPT results in incorrect max_val"
    assert (result.min_vals == tensor3.min_vals + tensor1.min_vals).all(), "SEPT + SEPT results in incorrect min_val"
    return None


# TODO: Fix REPT.min_vals and REPT.max_vals properties
def test_add_row_entities(row_data: List) -> None:
    """ Test normal addition of two REPTs"""
    tensor1 = REPT(rows=row_data)

    tensor2 = tensor1 + tensor1
    assert isinstance(tensor2, REPT), "Error: REPT + REPT != REPT "
    assert tensor2.shape == tensor1.shape, "Error: REPT + REPT changed shape of resultant REPT"
    assert tensor2 == tensor1 + tensor1, "Error: REPT + REPT failed"
    # assert (tensor2.min_vals == 2 * tensor1.min_vals), "REPT + REPT results in incorrect min_vals"
    # assert (tensor2.max_vals == 2 * tensor1.max_vals), "REPT + REPT results in incorrect max_vals"

    random_offset = np.random.randint(low=10, high=100) * np.random.random()
    tensor3 = tensor2 + tensor2 + random_offset
    assert isinstance(tensor3, REPT), "Error: REPT + REPT != REPT "
    assert tensor3.shape == tensor2.shape, "Error: REPT + REPT changed shape of resultant REPT"
    assert tensor3 - tensor2 + tensor2 == random_offset
    # assert (tensor3.min_vals == 2 * tensor2.min_vals + random_offset), "REPT + REPT results in incorrect min_vals"
    # assert (tensor3.max_vals == 2 * tensor2.max_vals + random_offset), "REPT + REPT results in incorrect max_vals"
    return None


def test_add_sub_equivalence(row_data: List) -> None:
    """ Test to see if addition of -ve and subtraction of +ve produce the same results"""
    tensor1 = REPT(rows=row_data)
    tensor2 = tensor1 * 2
    assert (tensor2.shape == tensor1.shape), "REPTs initialized incorrectly"

    assert tensor1 - 5 == tensor1 + 5 * -1, "Addition of -ve != Subtraction of +ve"
    assert tensor2 - tensor1 == tensor2 + tensor1 * -1, "Addition of -ve != Subtraction of +ve"
