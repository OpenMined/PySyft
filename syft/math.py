import numpy as np

from .tensor import TensorBase
from .tensor import _ensure_tensorbase

__all__ = [
    'cumprod','cumsum','ceil','dot', 'matmul',
]


def dot(tensor1, tensor2):
    """Returns inner product of two tensors.

    N-dimensional tensors are flattened into 1-D vectors, therefore this method should only be used on vectors.
    """

    tensor1 = _ensure_tensorbase(tensor1)
    tensor2 = _ensure_tensorbase(tensor2)

    if tensor1.encrypted is True or tensor2.encrypted is True:
        return NotImplemented
    return np.vdot(tensor1.data, tensor2.data)


def matmul(tensor1, tensor2):
    """Performs matrix multiplication between two tensors.

    Exact behavior depends on the input tensors' dimensionality like so:
    * If both tensors are 1-dimensional, their dot product is returned.
    * If both tensors are 2-dimensional, their matrix-matrix product is returned.
    * If either tensor has dimensionality > 2, the last 2 dimensions are treated as matrices and multiplied.
    * If tensor1 is 1-dimensional, it is converted to a matrix by prepending a 1 to its dimensions.
      This prepended dimension is removed after the matrix multiplication.
    * If tensor2 is 1-dimensional, it is converted to a matrix by prepending a 1 to its dimensions.
      This prepended dimension is removed after the matrix multiplication.
    """

    tensor1 = _ensure_tensorbase(tensor1)
    tensor2 = _ensure_tensorbase(tensor2)

    if tensor1.encrypted is True or tensor2.encrypted is True:
        return NotImplemented

    if tensor1.dim() == 1 and tensor2.dim() == 1:
        return dot(tensor1, tensor2)
    else:
        return TensorBase(np.matmul(tensor1.data, tensor2.data))


def ceil(tensor):
    """
    Returns the ceilling input tensor,element wise .

    Ceilling of an input scalar is the smallest integer such as :
    for each floating pount number x : a >= x

    Behavior is independent of a tensor's shape.  

    :input: TensorBase tensor\n
    :return: TensorBase tensor    
    """

    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True :
        return NotImplemented
    return TensorBase(np.ceil(tensor.data))


def cumsum(tensor,dim=0):
    """
    Returns the cumulative sum of the elements along a given dimension

    **Parameters**:
    * TensorBase tensor
    * Dimension on which the operation is done

    **returns**  A new 1D Tensor holding the result
    """
    
    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(np.cumsum(tensor.data,dim))

def cumprod(tensor,dim=0):
    """
    Returns the cumulative product of the elements along a given axis

    **Parameters**:
    * TensorBase tensor
    * Dimension on which the operation is done

    **returns** A new 1D Tensor holding the result
    """

    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(np.cumprod(tensor.data,dim))

