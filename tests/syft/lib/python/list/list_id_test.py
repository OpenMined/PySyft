# syft absolute
from syft.lib.python.int import Int
from syft.lib.python.list import List

sy_list = List([Int(1), Int(2), Int(3)])
other_list = List([Int(4)])
other = Int(4)
py_list = [1, 2, 3]


def test_id_contains():
    sy_res = sy_list.__contains__(other)
    py_res = py_list.__contains__(other)
    assert py_res == sy_res
    assert sy_res.id != sy_list.id


def test_id_delitem():
    tmp_list = List([1, 2, 3])
    id = tmp_list.id
    del tmp_list[0]
    assert id == tmp_list.id


def test_id_eq():
    sy_res = sy_list == other_list
    py_res = py_list == other_list
    assert py_res == sy_res
    assert sy_res.id != sy_list.id


def test_id_ge():
    sy_res = sy_list >= other_list
    py_res = py_list >= other_list
    assert py_res == sy_res
    assert sy_res.id != sy_list.id


def test_id_gt():
    sy_res = sy_list > other_list
    py_res = py_list > other_list
    assert py_res == sy_res
    assert sy_res.id != sy_list.id


def test_id_iadd():
    sy_res = sy_list.__iadd__(other_list)
    py_res = py_list.__iadd__(other_list)
    assert py_res == sy_res
    assert sy_res.id == sy_list.id


def test_id_imul():
    sy_res = sy_list.__imul__(other)
    py_res = py_list.__imul__(other)
    assert py_res == sy_res
    assert sy_res.id == sy_list.id


def test_id_le():
    sy_res = sy_list <= other_list
    py_res = py_list <= other_list
    assert py_res == sy_res
    assert sy_res.id != sy_list.id


def test_id_len():
    sy_res = sy_list.__len__()
    py_res = py_list.__len__()
    assert py_res == sy_res
    assert sy_res.id != sy_list.id


def test_id_lt():
    sy_res = sy_list < other_list
    py_res = py_list < other_list
    assert py_res == sy_res
    assert sy_res.id != sy_list.id


def test_id_mul():
    sy_res = sy_list.__mul__(other)
    py_res = py_list.__mul__(other)
    assert py_res == sy_res
    assert sy_res.id != sy_list.id


def test_id_ne():
    sy_res = sy_list != other
    py_res = py_list != other
    assert py_res == sy_res
    assert sy_res.id != sy_list.id


def test_id_radd():
    sy_res = sy_list.__radd__(other_list)
    assert sy_res.id != sy_list.id


def test_id_rmul():
    sy_res = sy_list.__rmul__(other)
    assert sy_res.id
    assert sy_res.id != sy_list.id


def test_id_sizeof():
    sy_res = sy_list.__sizeof__()
    assert sy_res.id
    assert sy_res.id != sy_list.id


def test_id_append():
    tmp_list = List([1, 2, 3])
    tmp_id = tmp_list.id
    tmp_list.append(4)
    assert tmp_id == tmp_list.id


def test_id_clear():
    tmp_list = List([1, 2, 3])
    tmp_id = tmp_list.id
    tmp_list.clear()
    assert tmp_id == tmp_list.id


def test_id_copy():
    sy_res = sy_list.copy()
    py_res = py_list.copy()
    assert py_res == sy_res
    assert sy_res.id != sy_list.id


def test_id_count():
    sy_res = sy_list.count(other)
    py_res = py_list.count(other)
    assert py_res == sy_res
    assert sy_res.id != sy_list.id


def test_id_te():
    tmp_list = List([1, 2, 3])
    tmp_id = tmp_list.id
    tmp_list.extend(List([1, 2, 3]))
    assert tmp_list.id == tmp_id


def test_id_pop():
    tmp_list = List([1, 2, 3])
    tmp_id = tmp_list.id
    tmp_list.pop(0)
    assert tmp_list.id == tmp_id


def test_id_remove():
    tmp_list = List([1, 2, 3])
    tmp_id = tmp_list.id
    tmp_list.remove(3)
    assert tmp_list.id == tmp_id


def test_id_sort():
    tmp_list = List([1, 2, 3])
    tmp_id = tmp_list.id
    tmp_list.sort()
    assert tmp_list.id == tmp_id
