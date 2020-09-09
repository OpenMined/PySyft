import pytest
from syft.lib.python.int import Int
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
    sy_int = Int(42)
    py_int = 42
    sy_int_API = set(dir(sy_int))
    py_int_API = set(dir(py_int))

    assert len(py_int_API - sy_int_API) == 0
    # immutable opeartors on the ID
    assert len(sy_int_API - py_int_API) == 15


@pytest.mark.parametrize("op", binop)
@pytest.mark.parametrize("py_obj", [42, 42.5, "42", True, False, None,
                                    42 + 5j, [42], {42: 42}])
def test_api_int(op, py_obj):
    sy_int = Int(42)
    func_py = getattr(py_obj, op)
    func_sy = getattr(sy_int, op)
    pypy_err, sysy_err, pysy_err, sypy_err = None, None, None, None
    pypy, sysy, pysy, sypy = None, None, None, None

    try:
        pypy = func_py(py_obj)
    except Exception as e_pypy:
        pypy_err = str(e_pypy)

    try:
        sysy = func_sy(sy_int)
    except Exception as e_sysy:
        sysy_err = str(e_sysy)

    try:
        pysy = func_py(sy_int)
    except Exception as e_pysy:
        pysy_err = str(e_pysy)

    try:
        sypy = func_sy(py_obj)
    except Exception as e_sypy:
        sypy_err = str(e_sypy)

    if any([pypy_err, sysy_err, pysy_err, sypy_err]):
        assert pypy_err == sysy_err == pysy_err == sypy_err
    else:
        assert pypy  == sysy == pysy == sypy
