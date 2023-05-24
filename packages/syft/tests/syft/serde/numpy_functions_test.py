# third party
import numpy as np
import pytest

# syft absolute
from syft import ActionObject
from syft.serde.lib_permissions import *
from syft.serde.lib_service_registry import *
from syft.service.response import SyftAttributeError
from syft.service.response import SyftError

PYTHON_ARRAY = [0, 1, 1, 2, 2, 3]


@pytest.mark.parametrize(
    "func",
    "func_arguments",
    [
        ("array", PYTHON_ARRAY),
        ("linspace", "10,10,10"),
        ("arange", "5,10,2"),
        ("logspace", "0,2"),
        ("zeros", "(1,2)"),
        ("identity", "4"),
        ("unique", PYTHON_ARRAY),
        ("mean", PYTHON_ARRAY),
        ("median", PYTHON_ARRAY),
        # ("digitize", PYTHON_ARRAY),
        ("reshape", f"{PYTHON_ARRAY}, (1,0)"),
        ("squeeze", PYTHON_ARRAY),
        ("count_nonzero", PYTHON_ARRAY),
        ("argwhere", PYTHON_ARRAY),
        ("argmax", PYTHON_ARRAY),
        ("argmin", PYTHON_ARRAY),
        ("sort", list(reversed(PYTHON_ARRAY))),
        ("absolute", PYTHON_ARRAY),
        ("clip", f"{PYTHON_ARRAY}, 0, 2"),
        # ("where", PYTHON_ARRAY),
        # ("put", PYTHON_ARRAY),
        # ("intersect1d", PYTHON_ARRAY),
        # ("setdiff1d", PYTHON_ARRAY),
        # ("setxor1d", PYTHON_ARRAY),
        # ("hsplit", PYTHON_ARRAY),
        # ("vsplit", PYTHON_ARRAY),
        # ("hstack", PYTHON_ARRAY),
        # ("vstack", PYTHON_ARRAY),
        # ("allclose", PYTHON_ARRAY),
        # ("equal", PYTHON_ARRAY),
        # ("repeat", PYTHON_ARRAY),
        # ("std", PYTHON_ARRAY),
        # ("var", PYTHON_ARRAY),
        # ("percentile", PYTHON_ARRAY),
        # ("var", PYTHON_ARRAY),
        # np.unique(arr, return_counts=True) #throws error
        # Not Working
        ("min", PYTHON_ARRAY),
        ("max", PYTHON_ARRAY),
        ("min", PYTHON_ARRAY),
    ],
)
def test_numpy_functions(func, func_arguments, numpy_syft_instance):
    try:
        result = eval(f"np_sy.{func}({func_arguments})")
    except Exception as e:
        assert (
            e == SyftAttributeError
        ), f"Can not evalute function {func} with arguments {func_arguments}"
    else:
        original_result = eval(f"np.{func}({func_arguments})")
        assert result == original_result
        assert isinstance(result, ActionObject)
