# stdlib
from typing import Any

# third party
import pandas as pd
import pytest

# syft absolute
import syft as sy

inputs = [
    {"func": "__eq__", "args": (pd.Categorical(["a", "b", "c", "a"], ordered=True))},
    {"func": "__ge__", "args": (pd.Categorical(["a", "b", "c", "a"], ordered=True))},
    {"func": "__gt__", "args": (pd.Categorical(["a", "b", "c", "a"], ordered=True))},
    {"func": "__le__", "args": (pd.Categorical(["a", "b", "c", "a"], ordered=True))},
    {"func": "__lt__", "args": (pd.Categorical(["a", "b", "c", "a"], ordered=True))},
    {"func": "__ne__", "args": (pd.Categorical(["a", "b", "c", "a"], ordered=True))},
    {"func": "__getitem__", "args": (1)},
    {"func": "__setitem__", "args": (1, "a")},
    {"func": "add_categories", "args": (["d"]), "kwargs": {"inplace": True}},
    {"func": "add_categories", "args": (["d"])},
    {"func": "__len__", "args": ()},
    {"func": "tolist", "args": ()},
    {"func": "to_list", "args": ()},
    {"func": "value_counts", "args": ()},
    {"func": "unique", "args": ()},
    {"func": "argmax", "args": ()},
    {"func": "argmin", "args": ()},
    {"func": "as_ordered", "args": ()},
    {"func": "as_unordered", "args": ()},
    {"func": "copy", "args": ()},
    {"func": "describe", "args": ()},
    {"func": "dropna", "args": ()},
    {"func": "dtype", "args": ()},
    {"func": "mode", "args": ()},
    {"func": "nbytes", "args": ()},
    {"func": "ndim", "args": ()},
    {"func": "ordered", "args": ()},
    {"func": "ravel", "args": ()},
    {"func": "remove_categories", "args": (["a"])},
    {"func": "remove_categories", "args": ("a")},
    {"func": "remove_unused_categories", "args": ()},
    {"func": "shape", "args": ()},
    {"func": "shift", "args": (1)},
    {"func": "shift", "args": (2)},
    {"func": "sort_values", "args": ()},
    {"func": "view", "args": ()},
]

objects = [
    pd.Categorical(["a", "b", "c", "a"], ordered=False),
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

    # third party
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
    else:
        assert y == y_dash
