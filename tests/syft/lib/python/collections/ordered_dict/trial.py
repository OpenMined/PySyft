from tests.syft.lib.python.bool.refactor_pointer_test import test_pointer_objectives

from typing import Tuple
from syft.lib.python.bool import Bool
from syft.lib.python.dict import Dict
from syft.lib.python.float import Float
from syft.lib.python.int import Int


SyFalse = Bool(False)
SyTrue = Bool(True)
PyFalse = False
PyTrue = True


def test_int():
    assert SyFalse.__float__() == 0.0
    assert SyTrue.__float__() == 1.0
    assert int(SyFalse) is not SyFalse
    assert int(SyTrue) is not SyTrue


py_res = 100.2031
py_res / 100
py_res = int(py_res * 1000) / 1000
py_res
isinstance(py_res, float)
int(py_res * 1000) / 1000

objects = (3, 1, 2)
a, b, c = objects
a
b
c
isinstance(a, Tuple)
pprint()
