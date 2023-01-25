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
