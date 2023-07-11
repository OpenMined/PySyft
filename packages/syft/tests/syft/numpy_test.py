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


def test_exception1_conversion_deletes_action_object(mock_object: ActionObject) -> None:
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


def test_exception2_numpy_methods_that_return_tuples(mock_object: ActionObject) -> None:
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


def test_exception3_numpy_methods_returning_new_arrays(
    mock_object: ActionObject,
) -> None:
    """
    NumPy methods such as `np.pad()` that take an array and return a new array will not return an ActionObject
    even if the inputs are ActionObjects.

    Some methods that "extract" such as `np.diag()` also suffer from this issue.
    """

    result = np.pad(mock_object, pad_width=1)
    assert isinstance(result, ActionObject), "The Result is no longer an action object"

    result2 = np.diag(mock_object)
    assert isinstance(result2, ActionObject), "The Result is no longer an action object"


def test_exception4_nan_behaviour(mock_object: ActionObject) -> None:
    """
    NaNs have strange behaviour and our mock objects don't work as intended with them for some reason
    """

    array = ActionObject.from_obj(np.empty_like(mock_object) * np.nan)
    assert np.nan in array, "The array is full of NaNs but a NaN was not detected"


def test_exception5_metadata_is_action_object(mock_object: ActionObject) -> None:
    """
    Perhaps things like `.shape` shouldn't return ActionObjects at all?
    """

    assert isinstance(mock_object.shape, tuple)


def test_exception6_action_object_integers(mock_object: ActionObject) -> None:
    """
    In certain places it seems that ActionObjects aren't a perfect replacement for integers and such
    """
    _ = np.unravel_index(12, mock_object.shape)


def test_exception7_chaining_operations(mock_object: ActionObject) -> None:
    """
    In examples such as the one below, each of the operations works individually but when done
    all at once, we are rewarded with an error :):
    """

    mock_object - mock_object.mean()
    mock_object.std()

    (mock_object - mock_object.mean()) / mock_object.std()


def test_exception8_inplace_modifications_kill_kernel(
    mock_object: ActionObject,
) -> None:
    """
    All of the expressions below will indeed kill your kernel
    """

    mock_object += 1
    np.add(mock_object, mock_object, out=mock_object)
    np.sub(mock_object, mock_object, out=mock_object)
    np.multiply(mock_object, mock_object, out=mock_object)
    np.divide(mock_object, mock_object, out=mock_object)
    mock_object[mock_object > 0.5] *= 2


def test_exception9_untriggered_memory_errors() -> None:
    """
    If you run something like `np.random.rand(156816, 36, 53806)` you will be given a `MemoryError:
    Unable to allocate 2.21 TiB for an array with shape (156816, 36, 53806) and data type float64`

    but other places where NumPy doesn't check for preallocation or something won't trigger MemoryErrors
    and instead will just kill your kernel.
    """

    np.sum(range(int(1e20)))


def test_exception10_numpy_flags_and_settings(mock_object: ActionObject) -> None:
    """
    Currently we don't have the ability to modify any of the flags on a NumPyActionObject, though we can see them
    """

    assert mock_object.flags is not None
    mock_object.flags.writeable = False


def test_exception11_numpy_domain_node_permissions() -> None:
    """
    Giving the user the ability to import numpy from the domain node directly might be dangerous.
    """
    # syft absolute
    import syft as sy

    worker = sy.Worker()
    # syft absolute
    from syft.core.node.new.client import SyftClient

    client = SyftClient.from_node(worker).login(
        email="info@openmined.org", password="changethis"
    )

    np = client.numpy

    # this might be dangerous- may be other similar things
    np.errstate(all="ignore")


def test_exception12_custom_classes(mock_object) -> None:
    """
    NumPy subclasses are probably tricky to work with and will have a ton of edge cases
    """

    # this won't raise an Error, but instead will give the user a warning and incorrect results
    np.random.shuffle(mock_object)
    raise AssertionError()


def test_exception13_record_arrays() -> None:
    """
    Record Arrays don't seem to work well with ActionObjects
    """

    Z_data = np.array([("Hello", 2.5, 3), ("World", 3.6, 2)])
    Z = ActionObject.from_obj(Z_data)
    np.core.records.fromarrays(Z.T, names="col1, col2, col3", formats="S8, f8, i8")
