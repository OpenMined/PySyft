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
