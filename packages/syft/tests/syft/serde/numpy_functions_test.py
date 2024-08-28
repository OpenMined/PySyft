# third party
import numpy as np
import pytest

# syft absolute
from syft import ActionObject

# relative
from ...utils.custom_markers import FAIL_ON_PYTHON_3_12_REASON
from ...utils.custom_markers import PYTHON_AT_LEAST_3_12

PYTHON_ARRAY = [0, 1, 1, 2, 2, 3]
NP_ARRAY = np.array([0, 1, 1, 5, 5, 3])
NP_2dARRAY = np.array([[3, 4, 5, 2], [6, 7, 2, 6]])


@pytest.mark.parametrize(
    "func, func_arguments",
    [
        ("array", "[0, 1, 1, 2, 2, 3]"),
        ("linspace", "10,10,10"),
        ("arange", "5,10,2"),
        ("logspace", "0,2"),
        ("zeros", "(1,2)"),
        ("identity", "4"),
        ("unique", "[0, 1, 1, 2, 2, 3]"),
        ("mean", "[0, 1, 1, 2, 2, 3]"),
        ("median", "[0, 1, 1, 2, 2, 3]"),
        ("digitize", "[0, 1, 1, 2, 2, 3], [0,1,2,3]"),
        ("reshape", "[0, 1, 1, 2, 2, 3], (6,1)"),
        ("squeeze", "[0, 1, 1, 2, 2, 3]"),
        ("count_nonzero", "[0, 1, 1, 2, 2, 3]"),
        ("argwhere", "[0, 1, 1, 2, 2, 3]"),
        ("argmax", "[0, 1, 1, 2, 2, 3]"),
        ("argmin", "[0, 1, 1, 2, 2, 3]"),
        ("sort", "list(reversed([0, 1, 1, 2, 2, 3]))"),
        ("absolute", "[0, 1, 1, 2, 2, 3]"),
        ("clip", "[0, 1, 1, 2, 2, 3], 0, 2"),
        ("put", " np.array([[3, 4, 5, 2], [6, 7, 2, 6]]), [1,2], [7,8]"),
        ("intersect1d", "[0, 1, 1, 2, 2, 3], np.array([0, 1, 1, 5, 5, 3])"),
        ("setdiff1d", "[0, 1, 1, 2, 2, 3], np.array([0, 1, 1, 5, 5, 3])"),
        ("setxor1d", "[0, 1, 1, 2, 2, 3], np.array([0, 1, 1, 5, 5, 3])"),
        ("hstack", "([0, 1, 1, 2, 2, 3], np.array([0, 1, 1, 5, 5, 3]))"),
        ("vstack", "([0, 1, 1, 2, 2, 3], np.array([0, 1, 1, 5, 5, 3]))"),
        ("allclose", "[0, 1, 1, 2, 2, 3], np.array([0, 1, 1, 5, 5, 3]), 0.5"),
        ("equal", "[0, 1, 1, 2, 2, 3], [0, 1, 1, 2, 2, 3]"),
        ("repeat", "2023, 4"),
        ("std", "[0, 1, 1, 2, 2, 3]"),
        ("var", "[0, 1, 1, 2, 2, 3]"),
        ("percentile", "[0, 1, 1, 2, 2, 3], 2"),
        ("var", "[0, 1, 1, 2, 2, 3]"),
        ("amin", "[0, 1, 1, 2, 2, 3]"),  # alias for min not exist in Syft
        ("amax", "[0, 1, 1, 2, 2, 3]"),  # alias for max not exist in Syft
        ("where", "a > 5, a, -1"),  # required condition
        # Not Working
        pytest.param(
            "hsplit",
            "np.array([[3, 4, 5, 2], [6, 7, 2, 6]]), 4",
            marks=pytest.mark.xfail(
                raises=ValueError, reason="Value error insinde Syft"
            ),
        ),
        pytest.param(
            "vsplit",
            "np.array([[3, 4, 5, 2], [6, 7, 2, 6]]), 2",
            marks=pytest.mark.xfail(
                raises=ValueError, reason="Value error insinde Syft"
            ),
        ),
        pytest.param(
            "unique",
            "np.array([0, 1, 1, 5, 5, 3]), return_counts=True",
            marks=pytest.mark.xfail(
                raises=(ValueError, AssertionError),
                reason="Kwargs Can not be properly unpacked",
            ),
        ),
    ],
)
@pytest.mark.xfail(PYTHON_AT_LEAST_3_12, reason=FAIL_ON_PYTHON_3_12_REASON)
def test_numpy_functions(func, func_arguments, request):
    # the problem is that ruff removes the unsued variable,
    # but this test case np_sy and a are considered as unused, though used in the eval string
    exec("np_sy = request.getfixturevalue('numpy_syft_instance')")
    try:
        if func == "where":
            exec(
                "a = np.array([[3, 4, 5, 2], [6, 7, 2, 6]])"
            )  # this is needed for where test
        result = eval(f"np_sy.{func}({func_arguments})")

    except Exception as e:
        assert (
            e == AttributeError
        ), f"Can not evaluate {func}({func_arguments}) with {e}"
        print(e)
    else:
        original_result = eval(f"np.{func}({func_arguments})")

        assert np.all(result == original_result)
        assert isinstance(result, ActionObject)
