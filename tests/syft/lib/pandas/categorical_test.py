# stdlib
from typing import Any
from typing import Dict
from typing import List

# third party
import pandas as pd
import pytest

# syft absolute
import syft as sy


def get_permission(
    obj: Any, node: sy.VirtualMachine, client: sy.VirtualMachineClient
) -> None:
    remote_obj = node.store[obj.id_at_location]
    remote_obj.read_permissions[client.verify_key] = obj.id_at_location


inputs = [
    ("__len__", None),
    ("tolist", None),
    ("to_list", None),
    ("value_counts", None),
    ("unique", None),
    ("argmax", None),
    ("argmin", None),
    ("as_ordered", None),
    ("as_unordered", None),
    ("copy", None),
    ("describe", None),
    ("dropna", None),
    ("dtype", None),
    # ( "equals",None),
    ("mode", None),
    ("nbytes", None),
    ("ndim", None),
    ("ordered", None),
    ("ravel", None),
    ("remove_categories", ["a"]),
    ("remove_categories", "a"),
    ("remove_unused_categories", None),
    ("shape", None),
    ("shift", 1),
    ("shift", 2),
    ("sort_values", None),
    ("view", None),
]

objects = [
    pd.Categorical(["a", "b", "c", "a"], ordered=False),
    pd.Categorical(["a", "b", "c", "a"], ordered=True),
]


@pytest.mark.vendor(lib="pandas")
@pytest.mark.parametrize("test_object", objects)
@pytest.mark.parametrize("func,args", inputs)
def test_categorical_func(
    test_object: Any,
    func: str,
    args: Any,
    node: sy.VirtualMachine,
    client: sy.VirtualMachineClient,
) -> None:
    sy.load("pandas")

    # third party
    import pandas as pd

    x = test_object
    x_ptr = test_object.send(client)

    op = getattr(x, func)
    op_ptr = getattr(x_ptr, func)
    # if op is a method
    if callable(op):
        if args is not None:
            y = op(args)
            y_ptr = op_ptr(args)
        else:
            y = op()
            y_ptr = op_ptr()
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
