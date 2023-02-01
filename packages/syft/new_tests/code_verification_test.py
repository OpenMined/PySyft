# third party
import numpy as np
import pytest

# syft absolute
from syft.core.node.new.action_object import ActionObject


@pytest.fixture
def data1() -> ActionObject:
    """Returns an Action Object with a NumPy dataset with values between -1 and 1"""
    return ActionObject(syft_action_data=2 * np.random.rand(10, 10) - 1)


@pytest.fixture
def data2() -> ActionObject:
    """Returns an Action Object with a NumPy dataset with values between -1 and 1"""
    return ActionObject(syft_action_data=2 * np.random.rand(10, 10) - 1)


def test_add_private(data1: ActionObject, data2: ActionObject) -> None:
    """Test whether adding two ActionObjects produces the correct history hash"""
    result1 = data1 + data2
    result2 = data1 + data2
    result3 = data2 + data1

    assert result1.syft_history_hash == result2.syft_history_hash
    assert result3.syft_history_hash != result2.syft_history_hash


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


def test_args_kwargs_identical(data1: ActionObject) -> None:
    """Test that data.std(1) == data.std(axis=1) are the same"""
    result1 = data1.std(1)
    result2 = data1.std(axis=1)

    assert result1.syft_history_hash == result2.syft_history_hash


def test_trace_easy(data1: ActionObject) -> None:
    """Test that we can recreate the correct history hash using TraceMode"""
    result1 = data1.std(axis=1)
    trace_result = ActionObject(syft_result_obj=np.array([]), id=data1.id).std(axis=1)

    assert result1.syft_history_hash == trace_result.syft_history_hash
