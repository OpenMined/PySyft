# stdlib
from typing import Any

# third party
import pandas as pd
import pytest

# syft absolute
import syft as sy

inputs = [
    {
        "func": "__eq__",
        "args": ([pd.Categorical(["a", "b", "c", "a"], ordered=True)]),
        "kwargs": {},
    },
    {
        "func": "__ge__",
        "args": ([pd.Categorical(["a", "b", "c", "a"], ordered=True)]),
        "kwargs": {},
    },
    {
        "func": "__gt__",
        "args": ([pd.Categorical(["a", "b", "c", "a"], ordered=True)]),
        "kwargs": {},
    },
    {
        "func": "__le__",
        "args": ([pd.Categorical(["a", "b", "c", "a"], ordered=True)]),
        "kwargs": {},
    },
    {
        "func": "__lt__",
        "args": ([pd.Categorical(["a", "b", "c", "a"], ordered=True)]),
        "kwargs": {},
    },
    {
        "func": "__ne__",
        "args": ([pd.Categorical(["a", "b", "c", "a"], ordered=True)]),
        "kwargs": {},
    },
    # Results on data_len access Error
    pytest.param(
        {"func": "__len__", "args": (), "kwargs": {}},
        marks=pytest.mark.xfail(reason="np.ndarray not Implemented"),
    ),
    {"func": "__getitem__", "args": ([1]), "kwargs": {}},
    {"func": "__setitem__", "args": (1, "a"), "kwargs": {}},
    {"func": "add_categories", "args": (["e"]), "kwargs": {}},
    {"func": "add_categories", "args": (["d"]), "kwargs": {"inplace": True}},
    {"func": "argmax", "args": [], "kwargs": {}},
    {"func": "argmin", "args": [], "kwargs": {}},
    {"func": "argsort", "args": [], "kwargs": {}},
    {"func": "as_ordered", "args": [], "kwargs": {}},
    {"func": "as_unordered", "args": [], "kwargs": {}},
    {"func": "copy", "args": [], "kwargs": {}},
    {"func": "describe", "args": [], "kwargs": {}},
    {"func": "dropna", "args": [], "kwargs": {}},
    {
        "func": "equals",
        "args": [pd.Categorical(["a", "b", "c", "a"], ordered=True)],
        "kwargs": {},
    },
    pytest.param(
        {"func": "factorize", "args": [], "kwargs": {}},
        marks=pytest.mark.xfail,
    ),
    {"func": "fillna", "args": ["a"], "kwargs": {}},
    {
        "func": "from_codes",
        "args": [[1, 0, 2, -1]],
        "kwargs": {"categories": ["a", "b", "c", "e"]},
    },
    {
        "func": "is_dtype_equal",
        "args": [pd.Categorical(["a", "b", "c", "a"], ordered=True)],
        "kwargs": {},
    },
    {"func": "isin", "args": [["a", "b"]], "kwargs": {}},
    {"func": "isna", "args": [], "kwargs": {}},
    {"func": "isnull", "args": [], "kwargs": {}},
    # TODO: {"func": "map", "args": [], "kwargs": {}},
    {"func": "max", "args": [], "kwargs": {}},
    {"func": "memory_usage", "args": [], "kwargs": {}},
    {"func": "min", "args": [], "kwargs": {}},
    {"func": "mode", "args": (), "kwargs": {}},
    {"func": "notna", "args": (), "kwargs": {}},
    {"func": "notnull", "args": (), "kwargs": {}},
    {"func": "ravel", "args": (), "kwargs": {}},
    {"func": "remove_categories", "args": (["a"]), "kwargs": {}},
    {"func": "remove_categories", "args": ("a"), "kwargs": {}},
    {"func": "remove_unused_categories", "args": (), "kwargs": {}},
    {"func": "rename_categories", "args": [[0, 1, 2, 3]], "kwargs": {}},
    {"func": "reorder_categories", "args": [["a", "c", "d", "b"]], "kwargs": {}},
    {"func": "repeat", "args": [2], "kwargs": {}},
    {"func": "replace", "args": ["a", "e"], "kwargs": {}},
    {"func": "reshape", "args": [-1], "kwargs": {}},
    {"func": "searchsorted", "args": ["c"], "kwargs": {}},
    {"func": "set_categories", "args": [["a", "c", "d"]], "kwargs": {}},
    {"func": "set_ordered", "args": [False], "kwargs": {}},
    {"func": "shift", "args": ([1]), "kwargs": {}},
    {"func": "shift", "args": ([2]), "kwargs": {}},
    {"func": "sort_values", "args": (), "kwargs": {}},
    {"func": "take", "args": [[1]], "kwargs": {}},
    {"func": "take_nd", "args": [[1]], "kwargs": {}},
    pytest.param(
        {"func": "to_dense", "args": [], "kwargs": {}},
        marks=pytest.mark.xfail(reason="np.ndarray not Implemented"),
    ),
    {"func": "tolist", "args": (), "kwargs": {}},
    {"func": "to_list", "args": (), "kwargs": {}},
    pytest.param(
        {"func": "to_numpy", "args": [], "kwargs": {}},
        marks=pytest.mark.xfail(reason="np.ndarray not Implemented"),
    ),
    {"func": "tolist", "args": [], "kwargs": {}},
    {"func": "unique", "args": [], "kwargs": {}},
    {"func": "value_counts", "args": [], "kwargs": {}},
    {"func": "view", "args": [], "kwargs": {}},
    # ===== properites =====
    {"func": "T", "args": (), "kwargs": {}},
    {"func": "codes", "args": (), "kwargs": {}},
    {"func": "dtype", "args": (), "kwargs": {}},
    {"func": "nbytes", "args": (), "kwargs": {}},
    {"func": "ndim", "args": (), "kwargs": {}},
    {"func": "ordered", "args": (), "kwargs": {}},
    {"func": "shape", "args": (), "kwargs": {}},
]

objects = [
    # pd.Categorical(["a", "b", "c", "a"], ordered=False), # Results in Error in comparision func
    pd.Categorical(["a", "b", "c", "a"], ordered=True),
]


@pytest.mark.slow
@pytest.mark.vendor(lib="pandas")
@pytest.mark.parametrize("test_object", objects)
@pytest.mark.parametrize("inputs", inputs)
def test_categorical_func(
    test_object: Any,
    inputs: dict,
    node: sy.VirtualMachine,
    client: sy.VirtualMachineClient,
) -> None:
    sy.load("pandas")
    sy.load("numpy")

    # third party
    import numpy as np
    import pandas as pd

    x = test_object
    x_ptr = test_object.send(client)

    func = inputs["func"]
    args = inputs["args"]
    kwargs = inputs["kwargs"]

    op = getattr(x, func)
    op_ptr = getattr(x_ptr, func)
    # if op is a method
    if callable(op):
        if args is not None:
            y = op(*args, **kwargs)
            y_ptr = op_ptr(*args, **kwargs)
    # op is a property
    else:
        y = op
        y_ptr = op_ptr

    y_dash = y_ptr.get()
    print(type(y))

    if (
        isinstance(y, pd.Categorical)
        or isinstance(y, pd.DataFrame)
        or isinstance(y, pd.Series)
    ):
        assert y.equals(y_dash)
    elif isinstance(y, np.ndarray):
        assert (y == y_dash).all()
    else:
        assert y == y_dash
