import torch as th
import numpy as np
import syft as sy


def test_numpy_add():
    """
    Test basic NumpyTensor addition
    """

    x = sy.NumpyTensor(numpy_tensor=[[1, 2, 3, 4]])
    y = x + x
    assert (y.child.child == np.array([2, 4, 6, 8])).all()


def test_numpy_subtract():
    """
    Test basic NumpyTensor subtraction
    """

    x = sy.NumpyTensor(numpy_tensor=np.array([[1, 2, 3, 4]]))
    y = x - x
    assert (y.child.child == np.array([0, 0, 0, 0])).all()


def test_numpy_multiply():
    """
    Test basic NumpyTensor multiplication
    """

    x = sy.NumpyTensor(numpy_tensor=np.array([[1, 2, 3, 4]]))
    y = x * x
    assert (y.child.child == np.array([1, 4, 9, 16])).all()


def test_numpy_divide():
    """
    Test basic NumpyTensor division
    """

    x = sy.NumpyTensor(numpy_tensor=np.array([[1, 2, 3, 4]]))
    y = x / x
    assert (y.child.child == np.array([1, 1, 1, 1])).all()


def test_numpy_dot():
    """
    Test basic NumpyTensor dot product
    """
    x = sy.NumpyTensor(numpy_tensor=np.array([[1, 2, 3, 4]]))
    y = x.dot(x.transpose())
    assert (y.child.child == np.array([[30]])).all()


def test_numpy_mm():
    """
    Test basic NumpyTensor matrix multiply
    """
    x = sy.NumpyTensor(numpy_tensor=np.array([[1, 2, 3, 4]]))
    y = x.mm(x.transpose())
    assert (y.child.child == np.array([[30]])).all()


def test_numpy_mm2():
    """
    Test @ based NumpyTensor matrix multiply
    """
    x = sy.NumpyTensor(numpy_tensor=np.array([[1, 2, 3, 4]]))
    y = x @ (x.transpose())
    assert (y.child.child == np.array([[30]])).all()


def test_numpy_transpose():
    """
    Test basic NumpyTensor transpose
    """
    x = sy.NumpyTensor(numpy_tensor=np.array([[1, 2, 3, 4]]))
    y = x.transpose(0, 1)
    assert (y.child.child == np.array([[1], [2], [3], [4]])).all()


def test_numpy_casting():
    """
    This tests the ability to cast a data tensor to a tensor chain
    with an underlying Numpy representation.
    """

    out = th.tensor([1, 2, 23, 4]).numpy_tensor()
    assert (out.child.child == np.array([1, 2, 23, 4])).all()
