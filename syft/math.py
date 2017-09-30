"""
    Module math implements mathematical primitives for tensor objects
"""
import numpy as np

from .tensor import TensorBase
from .tensor import _ensure_tensorbase

__all__ = [
    'cumprod', 'cumsum', 'ceil', 'dot', 'floor', 'matmul', 'addmm', 'addcmul',
    'addcdiv', 'addmv', 'addbmm', 'baddbmm', 'sigmoid', 'unsqueeze', 'tanh', 'relu',
    'zeros', 'ones', 'rand', 'randn', 'mm'
]


def zeros(dim):
    """Returns a tensor of zeros"""
    return TensorBase(np.zeros(dim))


def ones(dim):
    """Returns a tensor of ones"""
    return TensorBase(np.ones(dim))


def rand(dim):
    """Returns a tensor with numbers initialized according to a uniform
    distribution from 0 to 1"""
    return TensorBase(np.random.rand(dim))


def randn(dim):
    """Returns a tensor with initial numbers sampled from a standard normal
    distribution"""
    return TensorBase(np.random.randn(dim))


def dot(tensor1, tensor2):
    """Returns inner product of two tensors.

    N-dimensional tensors are flattened into 1-D vectors, therefore this
    method should only be used on vectors.
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
    * If both tensors are 2-dimensional, their matrix-matrix product is
    returned.

    * If either tensor has dimensionality > 2, the last 2 dimensions are
    treated as matrices and multiplied.

    * If tensor1 is 1-dimensional, it is converted to a matrix by prepending
    a 1 to its dimensions. This prepended dimension is removed after the
    matrix multiplication.

    * If tensor2 is 1-dimensional, it is converted to a matrix by prepending
    a 1 to its dimensions. This prepended dimension is removed after the
    matrix multiplication.
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
    for each floating point number x : a >= x

    Behavior is independent of a tensor's shape.

    :input: TensorBase tensor\n
    :return: TensorBase tensor
    """

    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(np.ceil(tensor.data))


def floor(tensor):
    """
    Returns the floored input tensor,element wise.
    Floor of an input scalar is the largest integer such as:
    for each floating point number x : a <= x

    Behavior is independent of a tensor's shape
    :input: TensorBase tensor\n
    :return: TensorBase tensor of floored elements .
    """

    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(np.floor(tensor.data))


def cumsum(tensor, dim=0):
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
    return TensorBase(np.cumsum(tensor.data, dim))


def cumprod(tensor, dim=0):
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
    return TensorBase(np.cumprod(tensor.data, dim))


def sigmoid(tensor):
    """ Returns a new tensor holding element wise values of Sigmoid function
        Sigmoid(x) = 1 / 1+exp(-x)
    """
    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(1 / (1 + np.exp(np.array(-tensor.data))))


def tanh(tensor):
    """ Returns a new tensor holding element wise values of tanh function
        tanh(x) = (e^(x) - e^(-x))/(e^(x) + e^(-x))
    """
    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(np.tanh(np.array(tensor.data)))


def relu(tensor):
    """ Return relu function
    """
    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(np.maximum(0, tensor.data))


def addmm(tensor1, tensor2, mat, beta=1, alpha=1):
    """Performs ((Mat*Beta)+((Tensor1.Tensor2)*Alpha)) and  returns the
    result as a Tensor
        Tensor1.Tensor2 is performed as Matrix product of two array
        The behavior depends on the arguments in the following way.
        *If both tensors are 1-dimensional, their dot product is returned.
        *If both arguments are 2-D they are multiplied like conventional
        matrices.

        *If either argument is N-D, N > 2, it is treated as a stack of
        matrices residing in the last two indexes and broadcast
        accordingly.

        *If the first argument is 1-D, it is promoted to a matrix by
        prepending a 1 to its dimensions. After matrix multiplication
        the prepended 1 is removed.

        *If the second argument is 1-D, it is promoted to a matrix by
        appending a 1 to its dimensions. After matrix multiplication
        the appended 1 is removed.
        """
    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(tensor2)
    _ensure_tensorbase(mat)
    if tensor1.encrypted or tensor2.encrypted or mat.encrypted:
        return NotImplemented
    else:
        delta = (np.matmul(tensor1.data, tensor2.data))
        return TensorBase(np.array(((mat.data) * beta) + (delta * alpha)))


def addcmul(tensor1, tensor2, mat, value=1):
    """Performs the element-wise multiplication of tensor1 by tensor2,
    multiply the result by the scalar value and add it to mat."""
    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(tensor2)
    _ensure_tensorbase(mat)
    if tensor1.encrypted or tensor2.encrypted or mat.encrypted:
        return NotImplemented
    else:
        out = (mat.data) + ((tensor1.data * tensor2.data) * value)
        return TensorBase(out)


def addcdiv(tensor1, tensor2, mat, value=1):
    """Performs the element-wise division of tensor1 by tensor2, multiply
    the result by the scalar value and add it to mat."""
    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(tensor2)
    _ensure_tensorbase(mat)
    if tensor1.encrypted or tensor2.encrypted or mat.encrypted:
        return NotImplemented
    else:
        out = (mat.data) + ((tensor1.data / tensor2.data) * value)
        return TensorBase(out)


def addmv(tensor1, mat, vec, beta=1, alpha=1):
    """"Performs a matrix-vector product of the matrix mat and the vector vec.
    The vector tensor is added to the final result.
          tensor1 and vec are 1d tensors
          out=(beta∗tensor)+(alpha∗(mat@vec2))"""
    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(vec)
    _ensure_tensorbase(mat)
    if vec.data.ndim != 1:
        print("dimension of vec is not 1")
    elif tensor1.data.ndim != 1:
        print("dimension of vec is not 1")
    elif tensor1.encrypted or vec.encrypted or mat.encrypted:
        return NotImplemented
    else:
        out = (tensor1.data * beta) + (np.matmul(mat.data, vec.data) * alpha)
        return TensorBase(out)


def addbmm(tensor1, tensor2, mat, beta=1, alpha=1):
    """Performs a batch matrix-matrix product of matrices stored in
    batch1(tensor1) and batch2(tensor2),
     with a reduced add step (all matrix multiplications get accumulated along
     the first dimension).
     mat is added to the final result.
     res=(beta∗M)+(alpha∗sum(batch1i@batch2i, i=0, b))
    * batch1 and batch2 must be 3D Tensors each containing the same number of
    matrices."""
    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(tensor2)
    _ensure_tensorbase(mat)
    if tensor2.data.ndim != 3:
        print("dimension of tensor2 is not 3")
    elif tensor1.data.ndim != 3:
        print("dimension of tensor1 is not 3")
    elif tensor1.encrypted or tensor2.encrypted or mat.encrypted:
        return NotImplemented
    else:
        mmul = np.matmul(tensor1.data, tensor2.data)
        sum_ = 0  # sum is a built in python function
        for i, _ in enumerate(mmul):
            sum_ += mmul[i]
        out = (mat.data * beta) + (alpha * sum_)
        return TensorBase(out)


def baddbmm(tensor1, tensor2, mat, beta=1, alpha=1):
    """Performs a batch matrix-matrix product of matrices in batch1(tensor1)
    and batch2(tensor2). mat is added to the final result.
      resi=(beta∗Mi)+(alpha∗batch1i×batch2i)
      *batch1 and batch2 must be 3D Tensors each containing the same number
      of matrices."""
    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(tensor2)
    _ensure_tensorbase(mat)
    if tensor2.data.ndim != 3:
        print("dimension of tensor2 is not 3")
    elif tensor1.data.ndim != 3:
        print("dimension of tensor1 is not 3")
    elif mat.data.ndim != 3:
        print("dimension of mat is not 3")
    elif tensor1.encrypted or tensor2.encrypted or mat.encrypted:
        return NotImplemented
    else:
        mmul = np.matmul(tensor1.data, tensor2.data)
        out = (mat.data * beta) + (mmul * alpha)
        return TensorBase(out)


def transpose(tensor1, dim0, dim1):
    """
    Performs tensor transpose operation, tranposing dim0 and dim1.
    Returns a tranposed TensorBase.
    """
    tensor1 = _ensure_tensorbase(tensor1)
    num_dims = len(tensor1.data.shape)
    axes = list(range(num_dims))

    if dim0 >= num_dims:
        print("dimension 0 out of range")
    elif dim1 >= num_dims:
        print("dimension 1 out of range")
    elif tensor1.encrypted:
        raise NotImplemented
    else:
        axes[dim0] = dim1
        axes[dim1] = dim0
        return TensorBase(np.transpose(tensor1.data, axes=axes))


def unsqueeze(tensor1, dim):
    """
    Performs 'unsqueeze' operation, returning a new tensor with a dimension
    of size one inserted at the specified position.
    """
    tensor1 = _ensure_tensorbase(tensor1)
    num_dims = len(tensor1.data.shape)

    if dim >= num_dims or dim < 0:
        print("dimension out of range")
    elif tensor1.encrypted:
        raise NotImplemented
    else:
        return TensorBase(np.expand_dims(tensor1.data, dim))


def mm(tensor1, tensor2):
    """
    Performs a matrix multiplication of :attr:`tensor1` and :attr:`tensor2`.

    If :attr:`tensor1` is a `n x m` Tensor, :attr:`tensor2` is a `m x p` Tensor,
    output will be a `n x p` Tensor.

    Args:
        tensor1 (Tensor): First Tensor to be multiplied
        tensor2 (Tensor): Second Tensor to be multiplied"""

    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(tensor2)

    if tensor1.encrypted or tensor2.encrypted:
        return NotImplemented
    else:
        return TensorBase(np.array(np.matmul(tensor1.data, tensor2.data)))
