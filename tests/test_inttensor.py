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

def test_max_():
    data = np.array([4,0,6,-3,8,-2]).astype('int')
    compare_data = np.array([1,-2,2,-3,0,-1]).astype('int')
    tensor = IntTensor(data)
    compare_to = IntTensor(compare_data)
    test_value = tensor.max_(compare_to)
    actual_value = IntTensor(np.array([4,0,6,-3,8,-1]))
    assert(test_value.equal(actual_value))

def test_view():
    a = IntTensor(np.array([[[9, 3, 1, 0], [6, 8, 6, 6]], [[1, 6, 8, 6], [5, 0, 2, 0]]]))
    a_v = a.view(-1)
    a_t_ground = IntTensor(np.array([9, 3, 1, 0, 6, 8, 6, 6, 1, 6, 8, 6, 5, 0, 2, 0]))
    assert(a_v.equal(a_t_ground))
    b_v = a.view(8, 2)
    b_v_ground = IntTensor(np.array([[9, 3], [1, 0], [6, 8], [6, 6], [1, 6], [8, 6], [5, 0], [2, 0]]))
    assert(b_v.equal(b_v_ground))
    a.view_(4, -1, 2)
    c_v_ground = IntTensor(np.array([[[9, 3], [1, 0]], [[6, 8], [6, 6]], [[1, 6], [8, 6]], [[5, 0], [2, 0]]]))
    assert(a.equal(c_v_ground))

def test_unfold():
    a = IntTensor(np.array([[-1, 2, 3, 5], [0, 4, 6, 7], [10, 3, 2, -5]], dtype=np.int32))

    # Test1
    expected_a = IntTensor(np.array([[[-1, 2, 3, 5], [0, 4, 6, 7]], [[0, 4, 6, 7], [10, 3, 2, -5]]], dtype=np.int32))
    actual_a = a.unfold(0, 2, 1)
    assert(actual_a.equal(expected_a))

    # Test2
    expected_a = IntTensor(np.array([[[-1, 2, 3], [0, 4, 6], [10, 3, 2]],
        [[2, 3, 5], [4, 6, 7], [3, 2, -5]]], dtype=np.int32))
    actual_a = a.unfold(1, 3, 1)
    assert(actual_a.equal(expected_a))

    # Test3
    expected_a = IntTensor(np.array([[[-1, 2], [0, 4], [10, 3]], [[3, 5], [6, 7], [2, -5]]], dtype=np.int32))
    actual_a = a.unfold(1, 2, 2)
    assert(actual_a.equal(expected_a))

