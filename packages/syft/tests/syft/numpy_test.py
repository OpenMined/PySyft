# third party
import numpy as np
import pandas as pd
import pytest
import torch

# relative
from ....syft.src.syft.core.node.new.action_object import ActionObject


@pytest.fixture
def reference_data() -> np.ndarray:
    return np.random.rand(5, 5)


@pytest.fixture
def mock_object(reference_data: np.ndarray) -> ActionObject:
    return ActionObject.from_obj(reference_data)


def exception1_conversion_deletes_action_object(mock_object: ActionObject) -> None:
    """
    Upon calling a conversion like np.asarray() or pd.DataFrame(array) the action object is destroyed.
    We lose the ability to track history hashes, lineage IDs, etc.
    """

    np_result = np.asarray(mock_object)
    assert isinstance(
        np_result, ActionObject
    ), "The Mock Object is no longer an action object"

    pd_result = pd.DataFrame(mock_object)
    assert isinstance(
        pd_result, ActionObject
    ), "The Mock Object is no longer an action object"

    torch.Tensor(mock_object)
    assert isinstance(
        np_result, ActionObject
    ), "The Mock Object is no longer an action object"


def exception2_numpy_methods_that_return_tuples(mock_object: ActionObject) -> None:
    """
    When working with a NumPy method that returns a tuple, the resultant tuple is an ActionObject, as
    opposed to each element in the result being an ActionObject.

    This isn't necessarily the worst outcome, but lineage IDs, syft_parents, and history hashes of the
    elements are now unrelated and may make debugging more difficult.
    """

    result = np.nonzero(mock_object)

    # this works
    assert isinstance(result, ActionObject)
    assert isinstance(result[0], ActionObject)
    assert isinstance(result[1], ActionObject)

    assert result.syft_lineage_id == result[0].syft_lineage_id
    assert result.syft_history_hash == result[0].syft_history_hash
    assert result[0].syft_lineage_id == result[1].syft_lineage_id


def exception3_numpy_methods_returning_new_arrays(mock_object: ActionObject) -> None:
    """
    NumPy methods such as `np.pad()` that take an array and return a new array will not return an ActionObject
    even if the inputs are ActionObjects.

    Some methods that "extract" such as `np.diag()` also suffer from this issue.
    """

    result = np.pad(mock_object, pad_width=1)
    assert isinstance(result, ActionObject), "The Result is no longer an action object"

    result2 = np.diag(mock_object)
    assert isinstance(result2, ActionObject), "The Result is no longer an action object"


def exception4_nan_behaviour(mock_object: ActionObject) -> None:
    """
    NaNs have strange behaviour and our mock objects don't work as intended with them for some reason
    """

    array = ActionObject.from_obj(np.empty_like(mock_object) * np.nan)
    assert np.nan in array, "The array is full of NaNs but a NaN was not detected"


def exception5_metadata_is_action_object(mock_object: ActionObject) -> None:
    """
    Perhaps things like `.shape` shouldn't return ActionObjects at all?
    """

    assert isinstance(mock_object.shape, tuple)


def exception6_action_object_integers(mock_object: ActionObject) -> None:
    """
    In certain places it seems that ActionObjects aren't a perfect replacement for integers and such
    """
    _ = np.unravel_index(12, mock_object.shape)


def exception7_chaining_operations(mock_object: ActionObject) -> None:
    """
    In examples such as the one below, each of the operations works individually but when done
    all at once, we are rewarded with an error :):
    """

    mock_object - mock_object.mean()
    mock_object.std()

    (mock_object - mock_object.mean()) / mock_object.std()


def exception8_inplace_modifications_kill_kernel(mock_object: ActionObject) -> None:
    """
    All of the expressions below will indeed kill your kernel
    """

    mock_object += 1
    np.add(mock_object, mock_object, out=mock_object)
    np.sub(mock_object, mock_object, out=mock_object)
    np.multiply(mock_object, mock_object, out=mock_object)
    np.divide(mock_object, mock_object, out=mock_object)
    mock_object[mock_object > 0.5] *= 2
