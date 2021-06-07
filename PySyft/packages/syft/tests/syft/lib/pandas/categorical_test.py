# stdlib
from typing import Any

# third party
import pytest

# syft absolute
import syft as sy

pd = pytest.importorskip("pandas")
np = pytest.importorskip("np")
sy.load("pandas")
sy.load("numpy")


inputs = [
    pytest.param(
        {"func": "to_dense", "args": [], "kwargs": {}},
        marks=pytest.mark.xfail(reason="np.ndarray object dtype not Implemented"),
    ),
    pytest.param(
        {"func": "to_numpy", "args": [], "kwargs": {}},
        marks=pytest.mark.xfail(reason="np.ndarray object dtype not Implemented"),
    ),
    {"func": "sort_values", "args": [], "kwargs": {}},
]

objects = [
    pd.Categorical(["a", "b", "c", "a"]),
    pd.Categorical([1, 2, 3, 1], ordered=True),
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

    if (
        isinstance(y, pd.Categorical)
        or isinstance(y, pd.DataFrame)
        or isinstance(y, pd.Series)
    ):
        assert y.equals(y_dash)
    elif isinstance(y, np.ndarray):
        assert (y == y_dash).all()
    elif isinstance(y, tuple):
        assert (y[0] == y_dash[0]).all()
        assert (y[1] == y_dash[1]).all()
    else:
        assert y == y_dash
