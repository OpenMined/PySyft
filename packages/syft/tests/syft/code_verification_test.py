# third party
import numpy as np
import pytest

# syft absolute
from syft.service.action.action_data_empty import ActionDataEmpty
from syft.service.action.action_object import ActionObject
from syft.service.action.numpy import NumpyArrayObject


@pytest.fixture
def data1() -> ActionObject:
    """Returns an Action Object with a NumPy dataset with values between -1 and 1"""
    yield NumpyArrayObject.from_obj(2 * np.random.rand(10, 10) - 1)


@pytest.fixture
def data2() -> ActionObject:
    """Returns an Action Object with a NumPy dataset with values between -1 and 1"""
    yield NumpyArrayObject.from_obj(2 * np.random.rand(10, 10) - 1)


@pytest.fixture
def empty1(data1) -> ActionObject:
    """Returns an Empty Action Object corresponding to data1"""
    yield ActionObject.empty(syft_internal_type=np.ndarray, id=data1.id)


@pytest.fixture
def empty2(data1) -> ActionObject:
    """Returns an Empty Action Object corresponding to data2"""
    yield NumpyArrayObject.from_obj(ActionDataEmpty(), id=data2.id)


def test_add_private(data1: ActionObject, data2: ActionObject) -> None:
    """Test whether adding two ActionObjects produces the correct history hash"""
    result1 = data1 + data2
    result2 = data1 + data2
    result3 = data2 + data1

    assert result1.syft_history_hash == result2.syft_history_hash
    assert result3.syft_history_hash == result2.syft_history_hash


def test_op(data1: ActionObject, data2: ActionObject) -> None:
    """Ensure that using a different op will produce a different history hash"""
    result1 = data1 + data2
    result2 = data1 == data2

    assert result1.syft_history_hash != result2.syft_history_hash


def test_args(data1: ActionObject, data2: ActionObject) -> None:
    """Ensure that passing args results in different history hashes"""
    result1 = data1.std()
    result2 = data1.std(1)

    assert result1.syft_history_hash != result2.syft_history_hash

    result3 = data2 + 3
    result4 = data2 + 4
    assert result3.syft_history_hash != result4.syft_history_hash


def test_kwargs(data1: ActionObject) -> None:
    """Ensure that passing kwargs results in different history hashes"""
    result1 = data1.std()
    result2 = data1.std(axis=1)

    assert result1.syft_history_hash != result2.syft_history_hash


def test_trace_single_op(data1: ActionObject) -> None:
    """Test that we can recreate the correct history hash using TraceMode"""
    result1 = data1.std()
    trace_result = NumpyArrayObject.from_obj(ActionDataEmpty(), id=data1.id).std()

    assert result1.syft_history_hash == trace_result.syft_history_hash


def test_empty_arithmetic_hash(data1: ActionObject, empty1: ActionObject) -> None:
    """Test that we can recreate the correct hash history using Empty Objects"""
    result1 = data1 + data1
    result2 = empty1 + empty1

    assert result1.syft_history_hash == result2.syft_history_hash


def test_empty_multiple_operations(data1: ActionObject, empty1: ActionObject) -> None:
    """Test that EmptyActionObjects are good for multiple operations"""
    real_tuple = (20, 5)
    remote_tuple = ActionObject.from_obj(real_tuple)

    step1 = data1.transpose()
    step2 = step1.reshape(remote_tuple)
    step3 = step2.std()

    target_hash = step3.syft_history_hash
    assert target_hash is not None

    step1 = empty1.transpose()
    step2 = step1.reshape(remote_tuple)
    step3 = step2.std()

    result_hash = step3.syft_history_hash
    assert result_hash is not None

    assert target_hash == result_hash


def test_history_hash_reproducibility(data1: ActionObject) -> None:
    """Test that the same history hash is produced, given the same inputs"""
    result1 = data1.mean().std()
    result2 = data1.mean().std()
    assert result1.syft_history_hash == result2.syft_history_hash

    remote_0 = ActionObject.from_obj(0)
    remote_10 = ActionObject.from_obj(10)

    mask = data1 > remote_0
    amount = data1 * remote_10
    result3 = mask * amount
    result4 = (data1 > remote_0) * (data1 * remote_10)
    assert result3.syft_history_hash == result4.syft_history_hash


def test_empty_action_obj_hash_consistency(
    data1: ActionObject, empty1: ActionObject
) -> None:
    """Test that Empty Action Objects and regular Action Objects can work together"""

    result1 = data1 + empty1
    result2 = data1 + data1

    assert result1.syft_history_hash == result2.syft_history_hash
