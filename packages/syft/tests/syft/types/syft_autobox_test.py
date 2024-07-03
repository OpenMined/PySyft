# stdlib
from contextlib import contextmanager

# third party
import numpy as np
import numpy.ctypeslib as npct
import pandas as pd
import pytest

# syft absolute
from syft.service.action.action_object import ActionObject
from syft.types.syft_autobox import Box
from syft.types.uid import UID


@contextmanager
def does_not_raise():
    yield


def test_autobox_string() -> None:
    b = "test"

    # a syft autoboxed string
    a = Box(b)
    assert isinstance(a, str) and isinstance(b, str)
    assert issubclass(type(a), str) and issubclass(type(b), str)
    assert type(b) in type(a).mro()


def test_autobox_numpy_array() -> None:
    arr = np.array([1, 2, 3, 4, 5])
    assert np.sum(arr) == 15

    arr_box = Box(arr)
    assert np.sum(arr_box) == 15

    # action object still works
    arr_ao = ActionObject.from_obj(arr)
    assert np.sum(arr_ao) == 15

    # action object can't work here
    with pytest.raises(TypeError, match="Cannot interpret.*"):
        npct.as_ctypes(arr_ao)

    # box can
    with does_not_raise():
        assert "c_long_Array_5" in str(npct.as_ctypes(arr_box))


def test_autobox_action_object() -> None:
    arr = np.array([1, 2, 3, 4, 5])
    arr_ao = ActionObject.from_obj(arr)
    assert isinstance(arr_ao.id, UID)

    arr_box = Box(arr_ao)
    assert isinstance(arr_box._syft_value, np.ndarray)
    assert arr_box._syft_uid == arr_ao.id


def test_autobox_pandas() -> None:
    data = {
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "City": ["New York", "Los Angeles", "Chicago"],
    }

    # normal df
    df = pd.DataFrame(data)

    # df from autoboxed dict
    auto_dict = Box(data)
    df2 = pd.DataFrame(auto_dict)

    assert all(df == df2)

    # make a boxed df
    df_ob = Box(df2)
    assert df_ob.Age.sum() == df.Age.sum()

    # can still init dataframes from boxed dataframes
    df3 = pd.DataFrame(df_ob)
    assert all(df == df3)
