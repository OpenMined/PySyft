# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd
import pytest

# syft absolute
from syft.service.action.action_data_empty import ActionDataEmpty
from syft.service.action.action_types import action_type_for_type
from syft.service.action.action_types import action_types


@pytest.mark.parametrize(
    "obj",
    [
        1,
        "str",
        2.3,
        False,
        [1, 2, 3],
        (1, 2, 3),
        {"a": 1, "b": 2},
        {1, 2, 3},
        ActionDataEmpty(),
    ],
)
def test_action_type_for_type_any(obj: Any):
    assert Any in action_types
    assert action_type_for_type(obj) == action_types[Any]


@pytest.mark.parametrize(
    "np_type",
    [
        np.ndarray,
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    ],
)
def test_action_type_for_type_numpy(np_type: Any):
    assert np_type in action_types

    if np_type == np.ndarray:
        np_obj = np.asarray([1, 2, 3])
    else:
        np_obj = np_type(1)

    assert action_type_for_type(np_obj) == action_types[np_type]


@pytest.mark.parametrize(
    "pd_type",
    [
        pd.Series,
        pd.DataFrame,
    ],
)
def test_action_type_for_type_pandas(pd_type: Any):
    assert pd_type in action_types
    if pd_type == pd.DataFrame:
        pd_obj = pd_type(np.asarray([[1, 2, 3]]))
    elif pd_type == pd.Series:
        pd_obj = pd_type(np.asarray([1, 2, 3]))
    else:
        raise RuntimeError(f"unhandled type {pd_type}")
    assert action_type_for_type(pd_obj) == action_types[pd_type]
