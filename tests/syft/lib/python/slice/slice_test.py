# syft absolute
from syft.lib.python.list import List
from syft.lib.python.slice import Slice
from syft.lib.python.string import String
from syft.lib.python.tuple import Tuple


def test_slice_types() -> None:
    py_string = "Python"
    py_list = ["P", "y", "t", "h", "o", "n"]
    py_tuple = ("P", "y", "t", "h", "o", "n")

    sy_string = String(py_string)
    sy_tuple = Tuple(py_tuple)
    sy_list = List(py_list)

    py_slice1 = slice(1)
    sy_slice1 = Slice(1)

    assert py_slice1.start == sy_slice1.start
    assert py_slice1.stop == sy_slice1.stop
    assert py_slice1.step == sy_slice1.step

    assert py_slice1 == sy_slice1

    py_slice2 = slice(1, 2)
    sy_slice2 = Slice(1, 2)

    assert py_slice2 == sy_slice2

    assert py_slice2.start == sy_slice2.start
    assert py_slice2.stop == sy_slice2.stop
    assert py_slice2.step == sy_slice2.step

    py_slice3 = slice(1, 2, -1)
    sy_slice3 = Slice(1, 2, -1)

    assert py_slice3 == sy_slice3

    assert py_slice3.start == sy_slice3.start
    assert py_slice3.stop == sy_slice3.stop
    assert py_slice3.step == sy_slice3.step

    assert sy_string[sy_slice1] == py_string[py_slice1]
    assert sy_string[sy_slice2] == py_string[py_slice2]
    assert sy_string[sy_slice3] == py_string[py_slice3]

    assert sy_tuple[sy_slice1] == py_tuple[py_slice1]
    assert sy_tuple[sy_slice2] == py_tuple[py_slice2]
    assert sy_tuple[sy_slice3] == py_tuple[py_slice3]

    assert sy_list[sy_slice1] == py_list[py_slice1]
    assert sy_list[sy_slice2] == py_list[py_slice2]
    assert sy_list[sy_slice3] == py_list[py_slice3]

    assert sy_list[py_slice3] == py_list[py_slice3]
