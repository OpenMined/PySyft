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


def test_transpose():
    a = IntTensor(np.array([[1, 2], [3, 4]]))
    a_t = a.T()
    a_t_ground = IntTensor(np.array([[1, 3], [2, 4]]))
    assert(a_t.equal(a_t_ground))
    b = IntTensor(np.array([[1, 2, 3], [4, 5, 6]]))
    b_t = b.T()
    b_t_ground = IntTensor(np.array([[1, 4], [2, 5], [3, 6]]))
    assert(b_t.equal(b_t_ground))
    c = IntTensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    c_t = c.T(0, 1)
    c_t_ground = IntTensor(np.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]]]))
    assert (c_t.equal(c_t_ground))


def test_view():
    a = IntTensor(np.array([[[9, 3, 1, 0], [6, 8, 6, 6]], [[1, 6, 8, 6], [5, 0, 2, 0]]]))
    a_v = a.view(-1)
    a_t_ground = IntTensor(np.array([9, 3, 1, 0, 6, 8, 6, 6, 1, 6, 8, 6, 5, 0, 2, 0]))
    assert(a_v.equal(a_t_ground))
    b_v = a.view(8, 2)
    b_v_ground = IntTensor(np.array([[9, 3], [1, 0], [6, 8], [6, 6], [1, 6], [8, 6], [5, 0], [2, 0]]))
    assert(b_v.equal(b_v_ground))
    c_v = a.view(4, -1, 2)
    c_v_ground = IntTensor(np.array([[[9, 3], [1, 0]], [[6, 8], [6, 6]], [[1, 6], [8, 6]], [[5, 0], [2, 0]]]))
    assert(c_v.equal(c_v_ground))
