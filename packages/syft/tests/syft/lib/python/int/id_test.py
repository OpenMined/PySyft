# syft absolute
from syft.lib.python.int import Int

test_int = Int(10)
other = Int(2)

python_int = 10


def test_id_abs():
    res = test_int.__abs__()
    py_res = python_int.__abs__()

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_add():
    res = test_int.__add__(other)
    py_res = python_int.__add__(other)

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_and():
    res = test_int.__and__(other)
    py_res = python_int.__and__(other)

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_ceil():
    res = test_int.__ceil__()
    py_res = python_int.__ceil__()

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_divmod():
    q, r = test_int.__divmod__(other)

    assert q.id
    assert r.id


def test_id_eq():
    res = test_int.__eq__(other)
    py_res = python_int.__eq__(other)

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_float():
    res = test_int.__float__()
    py_res = python_int.__float__()

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_floor():
    res = test_int.__floor__()
    py_res = python_int.__floor__()

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_floordiv():
    res = test_int.__floordiv__(other)
    py_res = python_int.__floordiv__(other)

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_ge():
    res = test_int.__ge__(other)
    py_res = python_int.__ge__(other)

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_gt():
    res = test_int.__gt__(other)
    py_res = python_int.__gt__(other)

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_hash():
    res = test_int.__hash__()
    py_res = python_int.__hash__()

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_iadd():
    res = test_int.__iadd__(other)

    assert res.id
    assert res.id == test_int.id


def test_id_ifloordiv():
    res = test_int.__ifloordiv__(other)

    assert res.id
    assert res.id == test_int.id


def test_id_imod():
    res = test_int.__imod__(other)

    assert res.id
    assert res.id == test_int.id


def test_id_imul():
    res = test_int.__imul__(other)

    assert res.id
    assert res.id == test_int.id


def test_id_invert():
    res = test_int.__invert__()
    py_res = python_int.__invert__()

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_ipow():
    res = test_int.__ipow__(other)

    assert res.id
    assert res.id == test_int.id


def test_id_isub():
    res = test_int.__isub__(other)

    assert res.id
    assert res.id == test_int.id


def test_id_itruediv():
    res = test_int.__itruediv__(other)

    assert res.id
    assert res.id == test_int.id


def test_id_le():
    res = test_int.__le__(other)
    py_res = python_int.__le__(other)

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_lshift():
    res = test_int.__lshift__(other)
    py_res = python_int.__lshift__(other)

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_lt():
    res = test_int.__lt__(other)
    py_res = python_int.__lt__(other)

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_mod():
    res = test_int.__mod__(other)
    py_res = python_int.__mod__(other)

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_mul():
    res = test_int.__mul__(other)
    py_res = python_int.__mul__(other)

    assert res == py_res
    assert res.id
    assert res.id != test_int.id


def test_id_ne():
    res = test_int.__ne__(other)
    py_res = python_int.__ne__(other)

    assert res == py_res
    assert res.id
    assert res.id != test_int.id
