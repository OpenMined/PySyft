# syft absolute
from syft.lib.python.int import Int
from syft.lib.python.slice import Slice

sy_slice = Slice(1, 3)
py_slice = slice(1, 3)
other_slice = Slice(Int(4))
other = Int(4)
py_list = [1, 2, 3]


def test_id_eq():
    sy_res = sy_slice == other_slice
    py_res = sy_slice == other_slice
    assert py_res == sy_res
    assert sy_res.id != sy_slice.id
