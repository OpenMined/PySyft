from syft import IntTensor
import numpy as np


def test_equal():
    a = IntTensor(np.array([0, 0]).astype('int'))
    b = IntTensor(np.array([0, 0]).astype('int'))
    different_shape_tensor = IntTensor(np.array([0]).astype('int'))
    different_value_tensor = IntTensor(np.array([1, 1]).astype('int'))
    assert(a.equal(b))
    assert(b.equal(a))
    assert(not a.equal(different_shape_tensor))
    assert(not different_shape_tensor.equal(a))
    assert(not a.equal(different_value_tensor))
    assert(not different_value_tensor.equal(a))
