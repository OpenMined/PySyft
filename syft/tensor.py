import numpy as np
import syft

__all__ = [
    'dot', 'matmul',
]

def _ensure_ndarray(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    return arr


def dot(tensor1, tensor2):
    """Returns inner product of two tensors.

    N-dimensional tensors are flattened into 1-D vectors, therefore this method should only be used on vectors.
    """

    if tensor1.encrypted is True or tensor2.encrypted is True:
        return NotImplemented

    return TensorBase(np.vdot(tensor1.data, tensor2.data))


def matmul(tensor1, tensor2):
    """Performs matrix multiplication between two tensors.

    Exact behavior depends on the input tensors' dimensionality like so:
    * If both tensors are 1-dimensional, their dot product is returned.
    * If both tensors are 2-dimensional, their matrix-matrix product is returned.
    * If either tensor has dimensionality > 2, the last 2 dimensions are treated as matrices and multiplied.
    * If tensor1 is 1-dimensional, it is converted to a matrix by prepending a 1 to its dimensions. This prepended dimension is removed after the matrix multiplication.
    * If tensor2 is 1-dimensional, it is converted to a matrix by prepending a 1 to its dimensions. This prepended dimension is removed after the matrix multiplication.
    """
    if tensor1.encrypted is True or tensor2.encrypted is True:
        return NotImplemented

    if tensor1.dim() == 1 and tensor2.dim() == 1:
        return dot(tensor1, tensor2)
    else:
        return TensorBase(np.matmul(tensor1.data, tensor2.data))


class TensorBase(object):
    """
    A base tensor class that perform basic element-wise operation such as
    addition, subtraction, multiplication and division
    """

    def __init__(self, arr_like, encrypted=False):
        self.data = _ensure_ndarray(arr_like)
        self.encrypted = encrypted

    def __add__(self, arr_like):
        """Performs element-wise addition between two array like objects"""
        if self.encrypted:
            return NotImplemented

        arr_like = _ensure_ndarray(arr_like)
        return self.data + arr_like

    def __iadd__(self, arr_like):
        """Performs in place element-wise addition between two array like objects"""
        if self.encrypted:
            return NotImplemented

        arr_like = _ensure_ndarray(arr_like)
        self.data = self.data + arr_like
        return self.data

    def __sub__(self, arr_like):
        """Performs element-wise subtraction between two array like objects"""
        if self.encrypted:
            return NotImplemented

        arr_like = _ensure_ndarray(arr_like)
        return self.data - arr_like

    def __isub__(self, arr_like):
        """Performs in place element-wise subtraction between two array like objects"""
        if self.encrypted:
            return NotImplemented

        arr_like = _ensure_ndarray(arr_like)
        self.data = self.data - arr_like
        return self.data



    def dot(self, tensor):
        """Returns inner product of two tensors"""
        if self.encrypted:
            return NotImplemented

        return syft.dot(self, tensor)

    def __matmul__(self, arr_like):
        """Performs matrix multiplication between two tensors"""
        return syft.matmul(self, arr_like)


    def __mul__(self, arr_like):
        """Performs element-wise multiplication between two array like objects"""
        if self.encrypted:
            return NotImplemented

        arr_like = _ensure_ndarray(arr_like)
        return self.data * arr_like

    def __imul__(self, arr_like):
        """Performs in place element-wise multiplication between two array like objects"""
        if self.encrypted:
            return NotImplemented

        arr_like = _ensure_ndarray(arr_like)
        self.data = self.data * arr_like
        return self.data

    def __truediv__(self, arr_like):
        """Performs element-wise division between two array like objects"""
        if self.encrypted:
            return NotImplemented

        arr_like = _ensure_ndarray(arr_like)
        return self.data / arr_like

    def __itruediv__(self, arr_like):
        """Performs in place element-wise subtraction between two array like objects"""
        if self.encrypted:
            return NotImplemented

        arr_like = _ensure_ndarray(arr_like)
        self.data = self.data / arr_like
        return self.data

    def shape(self):
        """Returns a tuple of input array dimensions."""
        if self.encrypted:
            return NotImplemented

        return self.data.shape


    def dim(self):
        """Returns an integer of the number of dimensions of this tensor."""
        if self.encrypted:
            return NotImplemented

        return self.data.ndim


    def sum(self, dim=None):
        """Returns the sum of all elements in the input array."""
        if self.encrypted:
            return NotImplemented

        if dim is None:
            return self.data.sum()
        else:
            return self.data.sum(axis=dim)
