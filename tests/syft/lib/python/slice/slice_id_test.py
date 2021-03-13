# syft absolute
from syft.lib.python.list import List
from syft.lib.python.slice import Slice
from syft.lib.python.string import String
from syft.lib.python.tuple import Tuple

py_string = "Python"
py_list = ["P", "y", "t", "h", "o", "n"]
py_tuple = ("P", "y", "t", "h", "o", "n")
sy_string = String(py_string)
sy_list = List(py_list)
sy_tuple = Tuple(py_tuple)
sy_string_slice = sy_string[Slice(3)]
sy_list_slice = sy_list[Slice(3)]
sy_tuple_slice = sy_tuple[Slice(3)]
py_string_slice = py_string[slice(3)]
py_list_slice = py_list[slice(3)]
py_tuple_slice = py_tuple[slice(3)]
other_string_slice = sy_string[Slice(1)]
other_list_slice = sy_list[Slice(1)]
other_tuple_slice = sy_tuple[Slice(1)]


def test_id_eq():
    sy_list_res = sy_list_slice == other_list_slice
    py_list_res = sy_list_slice == other_list_slice
    assert py_list_res == sy_list_res
    assert sy_list_res.id != sy_list_slice.id
