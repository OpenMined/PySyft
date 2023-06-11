# third party
import numpy as np
import pytest

# syft absolute
import syft
from syft import ActionObject

# from syft.serde.lib_permissions import *
# from syft.serde.lib_service_registry import *
from syft.service.response import SyftAttributeError

PYTHON_ARRAY = [0, 1, 1, 2, 2, 3]
NP_ARRAY = np.array([0, 1, 1, 5, 5, 3])
NP_2dARRAY = np.array([[3, 4, 5, 2], [6, 7, 2, 6]])


@pytest.mark.parametrize(
    "func, func_arguments",
    [
        ("array", [0, 1, 1, 2, 2, 3]),
        ("linspace", "10,10,10"),
        ("arange", "5,10,2"),
        ("logspace", "0,2"),
        ("zeros", "(1,2)"),
        ("identity", "4"),
        ("unique", [0, 1, 1, 2, 2, 3]),
        ("mean", [0, 1, 1, 2, 2, 3]),
        ("median", [0, 1, 1, 2, 2, 3]),
        ("digitize", "[0, 1, 1, 2, 2, 3], [0,1,2,3]"),
        ("reshape", "[0, 1, 1, 2, 2, 3], (6,1)"),
        ("squeeze", [0, 1, 1, 2, 2, 3]),
        ("count_nonzero", [0, 1, 1, 2, 2, 3]),
        ("argwhere", [0, 1, 1, 2, 2, 3]),
        ("argmax", [0, 1, 1, 2, 2, 3]),
        ("argmin", [0, 1, 1, 2, 2, 3]),
        ("sort", list(reversed([0, 1, 1, 2, 2, 3]))),
        ("absolute", [0, 1, 1, 2, 2, 3]),
        ("clip", "[0, 1, 1, 2, 2, 3], 0, 2"),
        ("put", f"{NP_2dARRAY}, [1,2], [7,8]"),
        ("intersect1d", "[0, 1, 1, 2, 2, 3], np.array([0, 1, 1, 5, 5, 3])"),
        ("setdiff1d", "[0, 1, 1, 2, 2, 3], np.array([0, 1, 1, 5, 5, 3])"),
        ("setxor1d", "[0, 1, 1, 2, 2, 3], np.array([0, 1, 1, 5, 5, 3])"),
        ("hsplit", "np.array([[3, 4, 5, 2], [6, 7, 2, 6]]), 4"),
        ("vsplit", f"{NP_2dARRAY}, 2"),
        ("hstack", f"{PYTHON_ARRAY}, {NP_ARRAY}"),
        ("vstack", f"{PYTHON_ARRAY}, {NP_ARRAY}"),
        ("allclose", f"{PYTHON_ARRAY}, {NP_ARRAY}, 0.5"),
        ("equal", f"{PYTHON_ARRAY}, {NP_ARRAY}"),
        ("repeat", "2023, 4"),
        ("std", PYTHON_ARRAY),
        ("var", PYTHON_ARRAY),
        ("percentile", f"{PYTHON_ARRAY}, 2"),
        ("var", PYTHON_ARRAY),
        # np.unique(arr, return_counts=True) #throws error
        # Not Working
        # ("min", [0, 1, 1, 2, 2, 3]),
        # ("max", [0, 1, 1, 2, 2, 3]),
        # ("min", [0, 1, 1, 2, 2, 3]),
        # ("where", f"np.array([0, 1, 1, 2, 2, 3])"),  # required condition
    ],
)
def test_numpy_functions(func, func_arguments):
    syft.Worker(name="Command_center")
    try:
        result = eval(f"np_sy.{func}({func_arguments})")
    except Exception as e:
        assert (
            e == SyftAttributeError
        ), f"Can not evalute function {func} with arguments {func_arguments}"
        pass
    else:
        original_result = eval(f"np.{func}({func_arguments})")

        assert result == original_result
        assert isinstance(result, ActionObject)
