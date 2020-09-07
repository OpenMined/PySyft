# third party
import torch as th

# syft absolute
from syft.lib.python.list import List

t1 = th.tensor([1, 2])
t2 = th.tensor([1, 3])
t3 = th.tensor([4, 5, 6])
t4 = th.tensor([7])

l1 = List([t1, t2])
l2 = List([t3, t4])

python_l1 = [t1, t2]
python_l2 = [t3, t4]


def test_id_add() -> None:
    res = l1 + l2
    py_res = python_l1 + python_l2

    assert res == py_res
    assert res.id
    assert res.id != l1.id
    assert res.id != l2.id
