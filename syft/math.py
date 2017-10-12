# coding=utf-8
"""
    Module math implements mathematical primitives for tensor objects

    Note:The Documentation in this file follows the NumPy Doc. Style;
         Hence, it is mandatory that future docs added here
         strictly follow the same, to maintain readability and consistency
         of the codebase.

    NumPy Documentation Style-
        http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html

"""
import numpy as np

from .tensor import TensorBase
from .tensor import _ensure_tensorbase

__all__ = [
    'cumprod', 'cumsum', 'ceil', 'dot', 'floor', 'matmul', 'addmm', 'addcmul',
    'addcdiv', 'addmv', 'bmm', 'addbmm', 'baddbmm', 'sigmoid', 'unsqueeze',
    'tanh', 'relu', 'zeros', 'ones', 'rand', 'randn', 'mm', 'fmod', 'diag', 'lerp', 'renorm', 'numel'
]


def zeros(dim):
    """
    Returns a tensor of zeros

    Parameters
    ----------
    dim:

    Returns
    -------
    TensorBase:
        Output Tensor
    """
    return TensorBase(np.zeros(dim))


def ones(dim):
    """
    Returns a tensor of ones

    Parameters
    ----------
    dim:

    Returns
    -------
    TensorBase:
        Output Tensor
    """
    return TensorBase(np.ones(dim))


def rand(dim):
    """
    Returns a tensor with numbers initialized according to a uniform
    distribution from 0 to 1

    Parameters
    ----------
    dim:

    Returns
    -------
    TensorBase:
        Output Tensor
    """
    return TensorBase(np.random.rand(dim))


def randn(dim):
    """
    Returns a tensor with initial numbers sampled from a standard normal
    distribution

    Parameters
    ----------
    dim:

    Returns
    -------
    TensorBase:
        Output Tensor
    """
    return TensorBase(np.random.randn(dim))


def dot(tensor1, tensor2):
    """
    Returns inner product of two tensors.

    N-dimensional tensors are flattened into 1-D vectors, therefore this
    method should only be used on vectors.

    Parameters
    ----------
    tensor1: TensorBase
        Tensor to be multiplied

    tensor2: TensorBase
        Tensor to be multiplied with

    Returns
    -------
    ndarray:
        Output N-Dimensional Array
    """

    tensor1 = _ensure_tensorbase(tensor1)
    tensor2 = _ensure_tensorbase(tensor2)

    if tensor1.encrypted is True or tensor2.encrypted is True:
        return NotImplemented
    return np.vdot(tensor1.data, tensor2.data)


def diag(tensor, diagonal=0):
    """
    * Returns a new 2D square tensor with the elements of 1D input tensor as the diagonal.
    * Returns a new 1D tensor with diagonal elements of 2D input tensor.

    * Optional argument diagonal value is about which diagonal to consider,
    zero is for main, positive for upper and negative for below diagonal

    Parameters
    ----------
    tensor : TensorBase
        The first operand in the diag operation
    diagonal : Integer
        The second operand in the diag operation

    Returns
    -------
    TensorBase
        Computed tensor result for diag operation
    """
    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    dim = tensor.dim()
    if dim == 1:
        return TensorBase(np.diag(tensor.data, diagonal))
    elif dim == 2:
        return TensorBase(np.diagonal(tensor.data, diagonal))
    else:
        raise ValueError("Input must be 1- or 2-d tensor.")


def matmul(tensor1, tensor2):
    """
    Performs matrix multiplication between two tensors.

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

    Parameters
    ----------
    tensor1: TensorBase
        Tensor to be multiplied

    tensor2: TensorBase
        Tensor to be multiplied with

    Returns
    -------
    TensorBase:
        Output Tensor
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

    Parameters
    ----------
    tensor: TensorBase
        input Tensor

    Returns
    -------
    TensorBase:
        Output Tensor
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

    Parameters
    ----------
    tensor: TensorBase
        input Tensor

    Returns
    -------
    TensorBase:
        Output Tensor; floored values
    """
    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(np.floor(tensor.data))


def cumsum(tensor, dim=0):
    """
    Returns the cumulative sum of the elements along a given dimension

    Parameters
    ----------
    tensor: TensorBase
        input Tensor

    dim:
        Dimension on which the operation is done

    Returns
    -------
    TensorBase:
        Output Tensor; 1D Tensor
    """
    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(np.cumsum(tensor.data, dim))


def cumprod(tensor, dim=0):
    """
    Returns the cumulative product of the elements along a given axis

    Parameters
    ----------
    tensor: TensorBase
        input Tensor

    dim:
        Dimension on which the operation is done

    Returns
    -------
    TensorBase:
        Output Tensor; 1D Tensor
    """
    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(np.cumprod(tensor.data, dim))


def sigmoid(tensor):
    """
    Returns a new tensor holding element wise values of Sigmoid function
    Sigmoid(x) = 1 / 1+exp(-x)

    Parameters
    ----------
    tensor: TensorBase
        input Tensor

    Returns
    -------
    TensorBase:
        Output Tensor;
    """
    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(1 / (1 + np.exp(np.array(-tensor.data))))


def tanh(tensor):
    """
    Returns a new tensor holding element wise values of tanh function
    tanh(x) = (e^(x) - e^(-x))/(e^(x) + e^(-x))

    Parameters
    ----------
    tensor: TensorBase
        input Tensor

    Returns
    -------
    TensorBase:
        Output Tensor;
    """
    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(np.tanh(np.array(tensor.data)))


def relu(tensor):
    """
    Return relu function

    Parameters
    ----------
    tensor: TensorBase
        input Tensor

    Returns
    -------
    TensorBase:
        Output Tensor;
    """
    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted is True:
        return NotImplemented
    return TensorBase(np.maximum(0, tensor.data))


def addmm(tensor1, tensor2, mat, beta=1, alpha=1):
    """
    Performs ((Mat*Beta)+((Tensor1.Tensor2)*Alpha)) and  returns the
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

    Parameters
    ----------
    tensor1: TensorBase

    tensor2: TensorBase

    mat:
        Matrix to the operation

    beta: ,optional

    alpha: ,optional

    Returns
    -------
    TensorBase:
        Output Tensor
    """
    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(tensor2)
    _ensure_tensorbase(mat)
    if tensor1.encrypted or tensor2.encrypted or mat.encrypted:
        return NotImplemented
    else:
        delta = (np.matmul(tensor1.data, tensor2.data))
        return TensorBase(np.array((mat.data * beta) + (delta * alpha)))


def addcmul(tensor1, tensor2, mat, value=1):
    """
    Performs the element-wise multiplication of tensor1 by tensor2,
    multiply the result by the scalar value and add it to mat.

    Parameters
    ----------
    tensor1: TensorBase

    tensor2: TensorBase

    mat:
        Matrix to the operation

    value: ,optional

    Returns
    -------
    TensorBase:
        Output Tensor
    """
    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(tensor2)
    _ensure_tensorbase(mat)
    if tensor1.encrypted or tensor2.encrypted or mat.encrypted:
        return NotImplemented
    else:
        out = mat.data + ((tensor1.data * tensor2.data) * value)
        return TensorBase(out)


def addcdiv(tensor1, tensor2, mat, value=1):
    """
    Performs the element-wise division of tensor1 by tensor2, multiply
    the result by the scalar value and add it to mat.

    Parameters
    ----------
    tensor1: TensorBase

    tensor2: TensorBase

    mat:
        Matrix to the operation

    value: ,optional

    Returns
    -------
    TensorBase:
        Output Tensor
    """
    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(tensor2)
    _ensure_tensorbase(mat)
    if tensor1.encrypted or tensor2.encrypted or mat.encrypted:
        return NotImplemented
    else:
        out = mat.data + ((tensor1.data / tensor2.data) * value)
        return TensorBase(out)


def addmv(tensor1, mat, vec, beta=1, alpha=1):
    """
    Performs a matrix-vector product of the matrix mat and the vector vec.

    The vector tensor is added to the final result.
    tensor1 and vec are 1d tensors
    out=(beta∗tensor)+(alpha∗(mat@vec2))

    Parameters
    ----------
    tensor1: TensorBase

    mat:
        Matrix for the operation

    vec:
        Vector

    beta: ,optional

    alpha: ,optional

    Returns
    -------
    TensorBase:
        Output Tensor
    """
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


def bmm(tensor1, tensor2):
    """
    Performs a batch matrix-matrix product of this tensor
    and tensor2. Both tensors must be 3D containing equal number
    of matrices.

    If this is a (b x n x m) Tensor, batch2 is a (b x m x p) Tensor,
    Result will be a (b x n x p) Tensor.

    Parameters
    ----------
    tensor1 : TensorBase
        The first operand in the bmm operation

    tensor2 : TensorBase
        The second operand in the bmm operation

    Returns
    -------
    TensorBase:
        Output Tensor; with bmm operation
    """
    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(tensor2)
    if tensor2.data.ndim != 3:
        print("dimension of tensor2 is not 3")
    elif tensor1.data.ndim != 3:
        print("dimension of tensor1 is not 3")
    elif tensor1.encrypted or tensor2.encrypted:
        return NotImplemented
    else:
        out = np.matmul(tensor1.data, tensor2.data)
        return TensorBase(out)


def addbmm(tensor1, tensor2, mat, beta=1, alpha=1):
    """
    Performs a batch matrix-matrix product of matrices stored in
    batch1(tensor1) and batch2(tensor2),
    with a reduced add step (all matrix multiplications get accumulated along
    the first dimension).
    mat is added to the final result.

    res=(beta∗M)+(alpha∗sum(batch1i@batch2i, i=0, b))

    batch1 and batch2 must be 3D Tensors each containing the same number of
    matrices.

    Parameters
    ----------
    tensor1: TensorBase

    tensor2: TensorBase

    mat:
        Matrix to the operation

    beta: ,optional

    alpha: ,optional

    Returns
    -------
    TensorBase:
        Output Tensor
    """
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
    """
    Performs a batch matrix-matrix product of matrices in batch1(tensor1)
    and batch2(tensor2). mat is added to the final result.

    resi=(beta∗Mi)+(alpha∗batch1i×batch2i)

    batch1 and batch2 must be 3D Tensors each containing the same number
    of matrices.

    Parameters
    ----------
    tensor1: TensorBase

    tensor2: TensorBase

    mat:
        Matrix to the operation

    beta: ,optional

    alpha: ,optional

    Returns
    -------
    TensorBase:
        Output Tensor
    """
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

    Parameters
    ----------
    tensor1: TensorBase

    dim0:
        Dimension 0

    dim1:
        Dimension 1

    Returns
    -------
    TensorBase:
        Output Tensor
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

    Parameters
    ----------
    tensor1: TensorBase

    dim:
        Dimension

    Returns
    -------
    TensorBase:
        Output Tensor
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

    Parameters
    ----------
    tensor1: TensorBase

    tensor2: TensorBase

    Returns
    -------
    TensorBase:
        Output Tensor
    """
    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(tensor2)

    if tensor1.encrypted or tensor2.encrypted:
        return NotImplemented
    else:
        return TensorBase(np.array(np.matmul(tensor1.data, tensor2.data)))


def fmod(tensor, divisor):
    """
    Performs the element-wise division of tensor by divisor.

    Parameters
    ----------
    tensor: TensorBase

    divisor: number or TensorBase

    Returns
    -------
    TensorBase:
        Output Tensor
    """
    if tensor.encrypted:
        return NotImplemented

    if isinstance(divisor, TensorBase):
        if divisor.encrypted:
            return NotImplemented
        divisor = divisor.data

    return TensorBase(np.fmod(tensor.data, divisor))


def numel(tensor):
    """
    Returns the total number of elements in the input Tensor.

    Parameters
    ----------

    Returns
    -------
    int:
        total number of elements in the input Tensor
    """
    if tensor.encrypted:
        return tensor.data.size
    else:
        return tensor.data.size


def lerp(tensor1, tensor2, weight):
    """
    Performs 'lerp' operation, returning a new tensor calculated by interpolation
    of two tensors using a weight.

    Parameters
    ----------
    tensor1: TensorBase
    tensor2: TensorBase

    weight:
        Weight supplied for iterpolation

    Returns
    -------
    TensorBase:
        Output Tensor
    """
    _ensure_tensorbase(tensor1)
    _ensure_tensorbase(tensor2)

    if tensor1.encrypted or tensor2.encrypted:
        return NotImplemented

    t1 = np.array(tensor1.data)
    t2 = np.array(tensor2.data)
    out = t1 + weight * (t2 - t1)
    return TensorBase(out)


def renorm(tensor1, p, dim, maxnorm):
    """
    Performs the scaling of elements along the dimension dim in tensor1 such that
    the p-norm of the sub-tensors along dim are less than or equal to maxnorm.
    Returns the result as an output tensor.

    The tensor, tensor1 is expected to have at least two dimesions, and the
    p-norm is defined to have powers greater than or equal to one.

    Parmeters
    ---------
    tensor1: TensorBase
        Input Tensor

    p:
        Power of the norm function

    dim:
        Dimension on which the operation is done

    maxnorm:
        Max value the p-norm is allowed to take on
    """
    tensor1 = _ensure_tensorbase(tensor1)
    dims = tensor1.data.ndim

    if tensor1.encrypted:
        return NotImplemented
    elif dims < 2:
        raise ValueError("tensor must have at least 2 dims")
    elif p < 1.0:
        raise ValueError("p must be a float greater than or equal to 1")
    else:
        # solve for c in maxnorm = sqrt(sum((c*x)**p))
        dim_2_sum = tuple(filter(lambda x: x != dim, range(dims)))
        norm = np.power(np.power(np.absolute(tensor1), p).sum(dim_2_sum), 1.0 / p)
        c = maxnorm / norm
        # only renorm when norm > maxnorm
        scalar = np.where(norm > maxnorm, c, 1)
        # broadcast along appropriate dim
        dim_array = np.ones((1, dims), int).ravel()
        dim_array[dim] = -1
        scalar_reshaped = scalar.reshape(dim_array)
        out = tensor1 * scalar_reshaped
    return TensorBase(out)
