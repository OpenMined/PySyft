import numpy as np
import syft

__all__ = [
    'equal', 'dot', 'matmul',
]

def _ensure_ndarray(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    return arr

def _ensure_tensorbase(tensor):
    if not isinstance(tensor, TensorBase):
        tensor = TensorBase(tensor)

    return tensor


def equal(tensor1, tensor2):
    """Checks if two tensors are equal.

    Two tensors are considered equal if they are the same size and contain the same elements.

    Assumption:
    tensor1 and tensor2 are of type TensorBase.
    Non-TensorBase objects will be converted to TensorBase objects.
    """

    tensor1 = _ensure_tensorbase(tensor1)
    tensor2 = _ensure_tensorbase(tensor2)

    if tensor1.encrypted is True or tensor2.encrypted is True:
        return NotImplemented

    return tensor1.data.shape == tensor2.data.shape and np.allclose(tensor1.data, tensor2.data)


def dot(tensor1, tensor2):
    """Returns inner product of two tensors.

    N-dimensional tensors are flattened into 1-D vectors, therefore this method should only be used on vectors.
    """

    tensor1 = _ensure_tensorbase(tensor1)
    tensor2 = _ensure_tensorbase(tensor2)

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

    tensor1 = _ensure_tensorbase(tensor1)
    tensor2 = _ensure_tensorbase(tensor2)

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

    def __add__(self, tensor):
        """Performs element-wise addition between two tensors"""
        if self.encrypted:
            return NotImplemented

        tensor = _ensure_tensorbase(tensor)
        return TensorBase(self.data + tensor.data)

    def __iadd__(self, tensor):
        """Performs in place element-wise addition between two tensors"""
        if self.encrypted:
            return NotImplemented

        tensor = _ensure_tensorbase(tensor)
        self.data = self.data + tensor.data
        return self

    def __sub__(self, tensor):
        """Performs element-wise subtraction between two tensors"""
        if self.encrypted:
            return NotImplemented

        tensor = _ensure_tensorbase(tensor)
        return TensorBase(self.data - tensor.data)

    def __isub__(self, tensor):
        """Performs in place element-wise subtraction between two tensors"""
        if self.encrypted:
            return NotImplemented

        tensor = _ensure_tensorbase(tensor)
        self.data = self.data - tensor.data
        return self

    def __eq__(self, tensor):
        """Checks if two tensors are equal"""
        if self.encrypted:
            return NotImplemented

        return syft.equal(self, tensor)

    def dot(self, tensor):
        """Returns inner product of two tensors"""
        if self.encrypted:
            return NotImplemented

        return syft.dot(self, tensor)

    def __matmul__(self, tensor):
        """Performs matrix multiplication between two tensors"""
        if self.encrypted:
            return NotImplemented

        return syft.matmul(self, tensor)

    def __mul__(self, tensor):
        """Performs element-wise multiplication between two tensors"""
        if self.encrypted:
            return NotImplemented

        tensor = _ensure_tensorbase(tensor)
        return TensorBase(self.data * tensor.data)

    def __imul__(self, tensor):
        """Performs in place element-wise multiplication between two tensors"""
        if self.encrypted:
            return NotImplemented

        tensor = _ensure_tensorbase(tensor)
        self.data = self.data * tensor.data
        return self

    def __truediv__(self, tensor):
        """Performs element-wise division between two tensors"""
        if self.encrypted:
            return NotImplemented

        tensor = _ensure_tensorbase(tensor)
        return TensorBase(self.data / tensor.data)

    def __itruediv__(self, tensor):
        """Performs in place element-wise subtraction between two tensors"""
        if self.encrypted:
            return NotImplemented

        tensor = _ensure_tensorbase(tensor)
        self.data = self.data / tensor.data
        return self

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
