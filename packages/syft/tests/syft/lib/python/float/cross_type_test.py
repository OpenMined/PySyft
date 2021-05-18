# third party
import pytest

# syft absolute
from syft.lib.python.float import Float

binop = [
    "__add__",
    "__and__",
    "__divmod__",
    "__eq__",
    "__floordiv__",
    "__ge__",
    "__gt__",
    "__le__",
    "__lshift__",
    "__lt__",
    "__mod__",
    "__mul__",
    "__ne__",
    "__or__",
    "__radd__",
    "__rand__",
    "__rdivmod__",
    "__rfloordiv__",
    "__rlshift__",
    "__rmod__",
    "__rmul__",
    "__ror__",
    "__rpow__",
    "__rrshift__",
    "__rshift__",
    "__rsub__",
    "__rtruediv__",
    "__rxor__",
    "__sub__",
    "__truediv__",
    "__xor__",
]


def test_api_sanity_check():
    sy_int = Float(42.5)
    py_int = 42.5
    sy_int_API = set(dir(sy_int))
    py_int_API = set(dir(py_int))
    print(sy_int_API - py_int_API)

    assert len(py_int_API - sy_int_API) == 0


@pytest.mark.parametrize("op", binop)
@pytest.mark.parametrize(
    "py_obj", [42, 42.0, "42", True, False, None, 42 + 5j, [42], {42: 42}]
)
def test_api_float(op, py_obj):
    sy_float = Float(42.5)
    py_float = 42.5

    try:
        func_py = getattr(py_float, op)
    except Exception:
        return

    func_sy = getattr(sy_float, op)

    pypy_err, sypy_err = None, None
    pypy, sypy = None, None

    try:
        pypy = func_py(py_obj)
    except Exception as e_pypy:
        pypy_err = str(e_pypy)

    try:
        sypy = func_sy(py_obj)
    except Exception as e_sysy:
        sypy_err = str(e_sysy)

    if any([pypy_err, sypy_err]):
        assert pypy_err == sypy_err
    else:
        assert pypy == sypy
