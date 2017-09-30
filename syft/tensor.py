# -*- coding: utf-8 -*-
import numpy as np
import syft
import scipy
import pickle

__all__ = [
    'equal', 'TensorBase',
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

    Two tensors are considered equal if they are the same size and contain the
    same elements.

    Assumption:
    tensor1 and tensor2 are of type TensorBase.
    Non-TensorBase objects will be converted to TensorBase objects.
    """

    tensor1 = _ensure_tensorbase(tensor1)
    tensor2 = _ensure_tensorbase(tensor2)

    if tensor1.encrypted or tensor2.encrypted:
        return NotImplemented

    left = tensor1.data.shape == tensor2.data.shape
    right = np.allclose(tensor1.data, tensor2.data)
    return left and right


class TensorBase(object):
    """
    A base tensor class that performs basic element-wise operation such as
    addition, subtraction, multiplication and division, and also dot and
    matrix products.
    """

    _mul_depth = 0
    _add_depth = 0

    def __init__(self, arr_like, encrypted=False):
        self.data = _ensure_ndarray(arr_like)
        self.encrypted = encrypted

    def _calc_mul_depth(self, tensor1, tensor2):
        if isinstance(tensor1, TensorBase) and isinstance(tensor2, TensorBase):
            self._mul_depth = max(tensor1._mul_depth, tensor2._mul_depth) + 1
        elif isinstance(tensor1, TensorBase):
            self._mul_depth = tensor1._mul_depth + 1
        elif isinstance(tensor2, TensorBase):
            self._mul_depth = tensor2._mul_depth + 1

    def _calc_add_depth(self, tensor1, tensor2):
        if isinstance(tensor1, TensorBase) and isinstance(tensor2, TensorBase):
            self._add_depth = max(tensor1._add_depth, tensor2._add_depth) + 1
        elif isinstance(tensor1, TensorBase):
            self._add_depth = tensor1._add_depth + 1
        elif isinstance(tensor2, TensorBase):
            self._add_depth = tensor2._add_depth + 1

    def encrypt(self, pubkey):
        """Encrypts the Tensor using a Public Key"""
        if self.encrypted:
            return NotImplemented
        else:
            if(type(pubkey) == syft.he.paillier.keys.PublicKey):
                out = syft.he.paillier.PaillierTensor(pubkey, self.data)
                return out
            else:
                return NotImplemented

    def decrypt(self, seckey):
        """Decrypts the tensor using a Secret Key"""
        if self.encrypted:
            return seckey.decrypt(self)
        else:
            return self

    def __len__(self):
        return len(self.data)

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
        self.data += tensor.data
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
        self.data -= tensor.data
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

        if tensor.encrypted:
            return tensor.dot(self)

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

        # if it's a sub-class of TensorBase, use the multiplication of that
        # subclass not this one.
        if(type(tensor) != TensorBase and isinstance(tensor, TensorBase)):
            return tensor * self
        else:
            tensor = _ensure_tensorbase(tensor)
            return TensorBase(tensor.data * self.data)

    def __imul__(self, tensor):
        """Performs in place element-wise multiplication between two tensors"""
        if self.encrypted:
            return NotImplemented

        if(type(tensor) != TensorBase and isinstance(tensor, TensorBase)):
            self.data = tensor.data * self.data
            self.encrypted = tensor.encrypted
        else:
            tensor = _ensure_tensorbase(tensor)
            self.data *= tensor.data
        return self

    def __truediv__(self, tensor):
        """Performs element-wise division between two tensors"""
        if self.encrypted:
            return NotImplemented

        if(type(tensor) != TensorBase and isinstance(tensor, TensorBase)):
            return NotImplemented  # it's not clear that this can be done
        else:
            tensor = _ensure_tensorbase(tensor)
            return TensorBase(self.data / tensor.data)

    def __itruediv__(self, tensor):
        """Performs in place element-wise subtraction between two tensors"""
        if self.encrypted:
            return NotImplemented

        tensor = _ensure_tensorbase(tensor)
        self.data = self.data / tensor.data
        return self

    def __setitem__(self, key, value):
        if(self.encrypted):
            return NotImplemented
        else:
            self.data[key] = value
            return self

    def __getitem__(self, position):
        """Get value at a specific index."""
        if self.encrypted:
            return NotImplemented
        else:
            out = self.data[position]
            if(len(self.shape()) == 1):
                return out
            else:
                return TensorBase(self.data[position], self.encrypted)

    def abs(self):
        """Returns absolute value of tensor as a new tensor"""
        if self.encrypted:
            return NotImplemented
        return np.absolute(self.data)

    def abs_(self):
        """Replaces tensor values with its absolute value"""
        if self.encrypted:
            return NotImplemented
        self.data = np.absolute(self.data)
        return self.data

    def shape(self):
        """Returns a tuple of input array dimensions."""
        if self.encrypted:
            return NotImplemented

        return self.data.shape

    def sqrt(self):
        """Returns the squared tensor."""
        if self.encrypted:
            return NotImplemented
        return np.sqrt(self.data)

    def sqrt_(self):
        """Inline squared tensor."""
        if self.encrypted:
            return NotImplemented
        self.data = np.sqrt(self.data)

    def dim(self):
        """Returns an integer of the number of dimensions of this tensor."""

        return self.data.ndim

    def sum(self, dim=None):
        """Returns the sum of all elements in the input array."""
        if self.encrypted:
            return NotImplemented

        if dim is None:
            return self.data.sum()
        else:
            return self.data.sum(axis=dim)

    def ceil(self):
        """Returns the ceilling of the input tensor elementwise."""
        if self.encrypted:
            return NotImplemented
        return syft.math.ceil(self.data)

    def ceil_(self):
        """Returns the ceilling of the input tensor elementwise."""
        if self.encrypted:
            return NotImplemented
        self.data = syft.math.ceil(self.data).data
        return self

    def floor_(self):
        """Inplace floor method"""
        if self.encrypted:
            return NotImplemented
        self.data = syft.math.floor(self.data).data
        return self

    def zero_(self):
        """Replaces tensor values with zeros"""
        if self.encrypted:
            return NotImplemented

        self.data.fill(0)
        return self.data

    def addmm(self, tensor2, mat, beta=1, alpha=1):
        """Performs ((Mat*Beta)+((Tensor1@Tensor2)*Alpha)) and  returns the
        result as a Tensor
            Tensor1.Tensor2 is performed as Matrix product of two array The
            behavior depends on the arguments in the following way.
            *If both tensors are 1-dimensional, their dot product is returned.
            *If both arguments are 2-D they are multiplied like conventional
            matrices.

            *If either argument is N-D, N > 2, it is treated as a stack of
            matrices residing in the last two indexes and broadcast
            accordingly.

            *If the first argument is 1-D, it is promoted to a matrix by
            prepending a 1 to its dimensions. After matrix multiplication the
            prepended 1 is removed.
            *If the second argument is 1-D, it is promoted to a matrix by
            appending a 1 to its dimensions. After matrix multiplication the
            appended 1 is removed.
            """
        return syft.addmm(self, tensor2, mat, beta, alpha)

    def addmm_(self, tensor2, mat, beta=1, alpha=1):
        """Performs ((Mat*Beta)+((Tensor1@Tensor2)*Alpha)) and updates Tensor1
        with result and reurns it
            Tensor1.Tensor2 is performed as Matrix product of two array The
            behavior depends on the arguments in the following way.

            *If both tensors are 1-dimensional, their dot product is returned.

            *If both arguments are 2-D they are multiplied like conventional
            matrices.

            *If either argument is N-D, N > 2, it is treated as a stack of
            matrices residing in the last two indexes and broadcast
            accordingly.

            *If the first argument is 1-D, it is promoted to a matrix by
            prepending a 1 to its dimensions. After matrix multiplication the
            prepended 1 is removed.

            *If the second argument is 1-D, it is promoted to a matrix by
            appending a 1 to its dimensions. After matrix multiplication the
            appended 1 is removed.
            """
        _ensure_tensorbase(tensor2)
        _ensure_tensorbase(mat)
        if self.encrypted or tensor2.encrypted or mat.encrypted:
            return NotImplemented
        else:
            self.data = np.array((np.matmul(self.data, tensor2.data)))
            self.data *= alpha
            mat.data *= beta
            self.data = self.data + mat.data
            return self

    def addcmul(self, tensor2, mat, value=1):
        """Performs the element-wise multiplication of tensor1 by tensor2,
        multiply the result by the scalar value and add it to mat."""
        return syft.addcmul(self, tensor2, mat, value)

    def addcmul_(self, tensor2, mat, value=1):
        """Performs implace element-wise multiplication of tensor1 by tensor2,
        multiply the result by the scalar value and add it to mat."""
        _ensure_tensorbase(tensor2)
        _ensure_tensorbase(mat)
        if self.encrypted or tensor2.encrypted or mat.encrypted:
            return NotImplemented
        else:
            self.data *= tensor2.data
            self.data *= value
            self.data += mat.data
            return self

    def addcdiv(self, tensor2, mat, value=1):
        """Performs the element-wise division of tensor1 by tensor2,
        multiply the result by the scalar value and add it to mat."""
        return syft.addcdiv(self, tensor2, mat, value)

    def addcdiv_(self, tensor2, mat, value=1):
        """Performs implace element-wise division of tensor1 by tensor2,
        multiply the result by the scalar value and add it to mat."""
        _ensure_tensorbase(tensor2)
        _ensure_tensorbase(mat)
        if self.encrypted or tensor2.encrypted or mat.encrypted:
            return NotImplemented
        else:
            self.data = self.data / tensor2.data
            self.data *= value
            self.data += mat.data
            return self

    def addmv(self, mat, vec, beta=1, alpha=1):
        """"Performs a matrix-vector product of the matrix mat and the vector
         vec. The vector tensor is added to the final result.
              tensor1 and vec are 1d tensors
              out=(beta∗tensor)+(alpha∗(mat@vec2))"""
        return syft.addmv(self, mat, vec, beta, alpha)

    def addmv_(self, mat, vec, beta=1, alpha=1):
        """"Performs a inplace matrix-vector product of the matrix mat and the
         vector vec. The vector tensor is added to the final result.
              tensor1 and vec are 1d tensors
              out=(beta∗tensor)+(alpha∗(mat@vec2))"""
        _ensure_tensorbase(vec)
        _ensure_tensorbase(mat)
        if vec.data.ndim != 1:
            print("dimension of vec is not 1")
        elif self.data.ndim != 1:
            print("dimension of tensor is not 1")
        elif self.encrypted or vec.encrypted or mat.encrypted:
            return NotImplemented
        else:
            self *= beta
            temp = np.matmul(mat.data, vec.data) * alpha
            self += temp
            return self

    def addbmm(self, tensor2, mat, beta=1, alpha=1):
        """Performs a batch matrix-matrix product of matrices stored in
        batch1(tensor1) and batch2(tensor2), with a reduced add step (all
        matrix multiplications get accumulated along the first dimension).
         mat is added to the final result.
         res=(beta∗M)+(alpha∗sum(batch1i@batch2i, i=0, b))
        * batch1 and batch2 must be 3D Tensors each containing the same
        number of matrices."""
        return syft.addbmm(self, tensor2, mat, beta, alpha)

    def addbmm_(self, tensor2, mat, beta=1, alpha=1):
        """Performs a inplace batch matrix-matrix product of matrices stored
        in batch1(tensor1) and batch2(tensor2), with a reduced add step
        (all matrix multiplications get accumulated along the first dimension).
         mat is added to the final result.
         res=(beta∗M)+(alpha∗sum(batch1i@batch2i, i=0, b)
        * batch1 and batch2 must be 3D Tensors each containing the same number
        of matrices.)"""
        _ensure_tensorbase(tensor2)
        _ensure_tensorbase(mat)
        if tensor2.data.ndim != 3:
            print("dimension of tensor2 is not 3")
        elif self.data.ndim != 3:
            print("dimension of tensor1 is not 3")
        elif self.encrypted or tensor2.encrypted or mat.encrypted:
            return NotImplemented
        else:
            self.data = np.matmul(self.data, tensor2.data)
            sum_ = 0  # sum is a python built in function a keyword !
            for i in range(len(self.data)):
                sum_ += self.data[i]
            self.data = (mat.data * beta) + (alpha * sum_)
            return self

    def baddbmm(self, tensor2, mat, beta=1, alpha=1):
        """Performs a batch matrix-matrix product of matrices in
        batch1(tensor1) and batch2(tensor2). mat is added to the final result.
          resi=(beta∗Mi)+(alpha∗batch1i×batch2i)
          *batch1 and batch2 must be 3D Tensors each containing the same number
          of matrices."""
        return syft.baddbmm(self, tensor2, mat, beta, alpha)

    def baddbmm_(self, tensor2, mat, beta=1, alpha=1):
        """Performs a batch matrix-matrix product of matrices in
        batch1(tensor1) and batch2(tensor2). mat is added to the final result.
          resi=(beta∗Mi)+(alpha∗batch1i×batch2i)
          *batch1 and batch2 must be 3D Tensors each containing the same number
          of matrices."""
        _ensure_tensorbase(tensor2)
        _ensure_tensorbase(mat)
        if tensor2.data.ndim != 3:
            print("dimension of tensor2 is not 3")
        elif self.data.ndim != 3:
            print("dimension of tensor1 is not 3")
        elif self.encrypted or tensor2.encrypted or mat.encrypted:
            return NotImplemented
        else:
            self.data = np.matmul(self.data, tensor2.data)
            self.data *= alpha
            self.data += (mat.data * beta)
            return self

    def max(self, axis=None):
        """ If axis is not specified, finds the largest element in the tensor. Otherwise, reduces along the specified axis.
        """
        if self.encrypted:
            return NotImplemented

        if axis is None:
            return _ensure_tensorbase(np.max(self.data))

        return _ensure_tensorbase(np.max(self.data, axis))

    def permute(self, dims):
        """
        Permute the dimensions of this tensor.
        Parameters:	*dims (int...) – The desired ordering of dimensions
        """
        if self.encrypted:
            return NotImplemented

        if dims is None:
            raise ValueError("dims cannot be none")

        return _ensure_tensorbase(np.transpose(self.data, dims))

    def transpose(self, dim0, dim1):
        """
        Returns the transpose along the dimensions in a new Tensor.
        """
        return syft.transpose(self.data, dim0, dim1)

    def transpose_(self, dim0, dim1):
        """
        Replaces the Tensor with its transpose along the dimensions.
        """
        num_dims = len(self.data.shape)
        axes = list(range(num_dims))

        if dim0 >= num_dims:
            print("dimension 0 out of range")
        elif dim1 >= num_dims:
            print("dimension 1 out of range")
        elif self.encrypted:
            raise NotImplemented
        else:
            axes[dim0] = dim1
            axes[dim1] = dim0
            self.data = np.transpose(self.data, axes=tuple(axes))

    def t(self):
        """
        Returns the transpose along dimensions 0, 1 in a new Tensor.
        """
        return self.transpose(0, 1)

    def t_(self):
        """
        Replaces the Tensor with its transpose along dimensions 0, 1.
        """
        self.transpose_(0, 1)

    def unsqueeze(self, dim):
        """
        Returns expanded Tensor. An additional dimension of size one is added
        to at index 'dim'.
        """
        return syft.unsqueeze(self.data, dim)

    def unsqueeze_(self, dim):
        """
        Replaces with an expanded Tensor. An additional dimension of size one
        is added to at index 'dim'.
        """
        num_dims = len(self.data.shape)

        if dim >= num_dims or dim < 0:
            print("dimension out of range")
        elif self.encrypted:
            raise NotImplemented
        else:
            self.data = np.expand_dims(self.data, dim)

    def exp(self):
        """Computes the exponential of each element in tensor."""
        if self.encrypted:
            return NotImplemented
        out = np.exp(self.data)
        return TensorBase(out)

    def exp_(self):
        """Computes the exponential of each element inplace."""
        if self.encrypted:
            return NotImplemented
        self.data = np.exp(self.data)
        return self

    def frac(self):
        """"Computes the fractional portion of each element in tensor."""
        if self.encrypted:
            return NotImplemented
        out = np.modf(self.data)[0]
        return TensorBase(out)

    def frac_(self):
        """"Computes the fractional portion of each element inplace."""
        if self.encrypted:
            return NotImplemented
        self.data = np.modf(self.data)[0]
        return self

    def sigmoid_(self):
        """
            Performs inline sigmoid function on the Tensor elementwise
            Implementation details:
            Because of the way syft.math.sigmoid operates on a Tensor Object
            calling it on self.data will cause an input error thus we call
            sigmoid on the tensor object and we take the member 'data' from the returned Tensor
        """
        if self.encrypted:
            return NotImplemented
        self.data = syft.math.sigmoid(self).data
        # self.data = np.array((1 / (1 + np.exp(np.array(-self.data)))))
        return self

    def tanh_(self):
        """
            Performs tanh (hyperbolic tangent) function on the Tensor elementwise
        """
        if self.encrypted:
            return NotImplemented
        self.data = syft.math.tanh(self).data
        # self.data = np.array(np.tanh(np.array(self.data)))
        return self

    def __str__(self):
        return "BaseTensor: " + str(self.data)

    def __repr__(self):
        return "BaseTensor: " + repr(self.data)

    def rsqrt(self):
        """Returns reciprocal of square root of Tensor element wise"""
        if self.encrypted:
            return NotImplemented
        out = 1 / np.sqrt(self.data)
        return TensorBase(out)

    def rsqrt_(self):
        """Computes reciprocal of square root of Tensor elements inplace"""
        if self.encrypted:
            return NotImplemented
        self.data = 1 / np.sqrt(self.data)

    def sign(self):
        """Return a tensor that contains sign of each element """
        if self.encrypted:
            return NotImplemented
        out = np.sign(self.data)
        return TensorBase(out)

    def sign_(self):
        """Computes the sign of each element of the Tensor inplace"""
        if self.encrypted:
            return NotImplemented
        self.data = np.sign(self.data)

    def to_numpy(self):
        """Returns the tensor as numpy.ndarray"""
        if self.encrypted:
            return NotImplemented
        return np.array(self.data)

    def reciprocal(self):
        """Computes element wise reciprocal"""
        if self.encrypted:
            return NotImplemented
        out = 1 / np.array(self.data)
        return TensorBase(out)

    def reciprocal_(self):
        """Computes element wise reciprocal"""
        if self.encrypted:
            return NotImplemented
        self.data = 1 / np.array(self.data)

    def log(self):
        """performs elementwise logarithm operation
        and returns a new Tensor"""
        if self.encrypted:
            return NotImplemented
        out = np.log(self.data)
        return TensorBase(out)

    def log_(self):
        """performs elementwise logarithm operation inplace"""
        if self.encrypted:
            return NotImplemented
        self.data = np.log(self.data)
        return self

    def log1p(self):
        """performs elementwise log(1+x) operation
        and returns new tensor"""
        if self.encrypted:
            return NotImplemented
        out = np.log1p(self.data)
        return TensorBase(out)

    def log1p_(self):
        """performs elementwise log(1+x) operation inplace"""
        if self.encrypted:
            return NotImplemented
        self.data = np.log1p(self.data)
        return self

    def log_normal_(self, mean=0, stdev=1.0):
        """Fills give tensor with samples from a lognormal distribution
        with given mean and stdev"""
        if self.encrypted:
            return NotImplemented
        self.data = np.random.lognormal(mean, stdev, self.shape())
        return self

    def clamp(self, minimum=None, maximum=None):
        """Returns a clamped tensor into the range [min, max], elementwise"""
        if self.encrypted:
            return NotImplemented
        return TensorBase(np.clip(self.data, a_min=minimum, a_max=maximum))

    def clamp_(self, minimum=None, maximum=None):
        """Clamp the tensor, in-place, elementwise into the range [min, max]"""
        if self.encrypted:
            return NotImplemented
        self.data = np.clip(self.data, a_min=minimum, a_max=maximum)
        return self

    def clone(self):
        """Returns a copy of the tensor. The copy has the same size and data type as the original tensor."""
        if self.encrypted:
            return NotImplemented
        return TensorBase(np.copy(self.data))

    def chunk(self, n, dim=0, same_size=False):
        """Returns a list of tensors by splitting the tensor into a number of chunks along a given dimension.
        Raises an exception if same_size is set to True and given tensor can't be split in n same-size chunks along dim."""
        if self.encrypted:
            return NotImplemented
        if same_size:
            return [TensorBase(x) for x in np.split(self.data, n, dim)]
        else:
            return [TensorBase(x) for x in np.array_split(self.data, n, dim)]

    def gt(self, other):
        """Returns a new Tensor having boolean True values where an element of the calling tensor is greater than the second Tensor, False otherwise.
        The second Tensor can be a number or a tensor whose shape is broadcastable with the calling Tensor."""
        other = _ensure_tensorbase(other)
        if self.encrypted or other.encrypted:
            return NotImplemented
        return TensorBase(np.greater(self.data, other.data))

    def gt_(self, other):
        """Writes in-place, boolean True values where an element of the calling tensor is greater than the second Tensor, False otherwise.
        The second Tensor can be a number or a tensor whose shape is broadcastable with the calling Tensor."""
        other = _ensure_tensorbase(other)
        if self.encrypted or other.encrypted:
            return NotImplemented
        self.data = np.greater(self.data, other.data)
        return self

    def lt(self, other):
        """Returns a new Tensor having boolean True values where an element of the calling tensor is less than the second Tensor, False otherwise.
        The second Tensor can be a number or a tensor whose shape is broadcastable with the calling Tensor."""
        other = _ensure_tensorbase(other)
        if self.encrypted or other.encrypted:
            return NotImplemented
        return TensorBase(np.less(self.data, other.data))

    def lt_(self, other):
        """Writes in-place, boolean True values where an element of the calling tensor is less than the second Tensor, False otherwise.
        The second Tensor can be a number or a tensor whose shape is broadcastable with the calling Tensor."""
        other = _ensure_tensorbase(other)
        if self.encrypted or other.encrypted:
            return NotImplemented
        self.data = np.less(self.data, other.data)
        return self

    def ge(self, other):
        """Returns a new Tensor having boolean True values where an element of the calling tensor is greater or equal than the second Tensor, False otherwise.
        The second Tensor can be a number or a tensor whose shape is broadcastable with the calling Tensor."""
        other = _ensure_tensorbase(other)
        if self.encrypted or other.encrypted:
            return NotImplemented
        return TensorBase(np.greater_equal(self.data, other.data))

    def ge_(self, other):
        """Writes in-place, boolean True values where an element of the calling tensor is greater or equal than the second Tensor, False otherwise.
        The second Tensor can be a number or a tensor whose shape is broadcastable with the calling Tensor."""
        other = _ensure_tensorbase(other)
        if self.encrypted or other.encrypted:
            return NotImplemented
        self.data = np.greater_equal(self.data, other.data)
        return self

    def le(self, other):
        """Returns a new Tensor having boolean True values where an element of the calling tensor is less or equal than the second Tensor, False otherwise.
        The second Tensor can be a number or a tensor whose shape is broadcastable with the calling Tensor."""
        other = _ensure_tensorbase(other)
        if self.encrypted or other.encrypted:
            return NotImplemented
        return TensorBase(np.less_equal(self.data, other.data))

    def le_(self, other):
        """Writes in-place, boolean True values where an element of the calling tensor is less or equal than the second Tensor, False otherwise.
        The second Tensor can be a number or a tensor whose shape is broadcastable with the calling Tensor."""
        other = _ensure_tensorbase(other)
        if self.encrypted or other.encrypted:
            return NotImplemented
        self.data = np.less_equal(self.data, other.data)
        return self

    def bernoulli(self, p):
        """
        Returns a Tensor filled with binary random numbers (0 or 1) from a bernoulli distribution
        with probability and shape specified by p(arr_like).

        The p Tensor should be a tensor containing probabilities to be used for drawing the
        binary random number. Hence, all values in p have to be in the range: 0<=p<=1
        """
        if self.encrypted:
            return NotImplemented
        p = _ensure_tensorbase(p)
        return TensorBase(np.random.binomial(1, p.data))

    def bernoulli_(self, p):
        """
        Fills the Tensor in-place with binary random numbers (0 or 1) from a bernoulli distribution
        with probability and shape specified by p(arr_like)

        The p Tensor should be a tensor containing probabilities to be used for drawing the
        binary random number. Hence, all values in p have to be in the range: 0<=p<=1
        """
        if self.encrypted:
            return NotImplemented
        p = _ensure_tensorbase(p)
        self.data = np.random.binomial(1, p.data)
        return self

    def uniform_(self, low=0, high=1):
        """Fills the tensor in-place with numbers sampled unifromly
        over the half-open interval [low,high) or from the uniform distribution"""
        if self.encrypted:
            return NotImplemented
        self.data = np.random.uniform(low=low, high=high, size=self.shape())
        return self

    def uniform(self, low=0, high=1):
        """Returns a new tensor filled with numbers sampled unifromly
        over the half-open interval [low,high) or from the uniform distribution"""
        if self.encrypted:
            return NotImplemented
        out = np.random.uniform(low=low, high=high, size=self.shape())
        return TensorBase(out)

    def fill_(self, value):
        """Fills the tensor in-place with the specified value"""
        if self.encrypted:
            return NotImplemented
        self.data.fill(value)
        return self

    def tolist(self):
        """Returns a new tensor as (possibly a nested) list"""
        if self.encrypted:
            return NotImplemented
        out = self.data.tolist()
        return out

    def topk(self, k, largest=True):
        """Returns a new tensor with the sorted k largest (or smallest) values"""
        if self.encrypted:
            return NotImplemented
        out_sort = np.sort(self.data)
        if self.data.ndim > 1:
            out = np.partition(out_sort, kth=k)
            out = out[:, -k:] if largest else out[:, :k]
        else:
            out = np.partition(out_sort, kth=k)
            out = out[-k:] if largest else out[:k]
        return TensorBase(out)

    def trace(self, axis1=None, axis2=None):
        """Returns a new tenosr with the sum along diagonals of a 2D tensor.
           Axis1 and Axis2 are used to extract 2D subarray for sum calculation
           along diagonals, if tensor has more than two dimensions. """
        if self.encrypted:
            return NotImplemented
        if axis1 is not None and axis2 is not None and self.data.ndim > 2:
            out = np.trace(a=self.data, axis1=axis1, axis2=axis2)
        else:
            out = np.trace(a=self.data)
        return TensorBase(out)

    def view(self, *args):
        """View the tensor."""
        if self.encrypted:
            return NotImplemented
        else:
            dt = np.copy(self.data)
            return TensorBase(dt.reshape(*args))

    def view_as(self, tensor):
        """ View as another tensor's shape """
        if self.encrypted:
            return NotImplemented
        else:
            return self.view(tensor.shape())

    def resize_(self, *size):
        input_size = np.prod(size)
        extension = input_size - self.data.size
        flattened = self.data.flatten()
        if input_size >= 0:
            if extension > 0:
                data = np.append(flattened, np.zeros(extension))
                self.data = data.reshape(*size)
                print(self.data)
            elif extension < 0:
                size_ = self.data.size + extension
                self.data = flattened[:size_]
                self.data = self.data.reshape(*size)
                print(self.data)
            else:
                self.data = self.data.reshape(*size)
                print(self.data)
        else:
            raise ValueError('negative dimension not allowed')

    def resize_as_(self, tensor):
        size = tensor.data.shape
        self.resize_(size)

    def round(self, decimals=0):
        """Returns a new tensor with elements rounded off to a nearest decimal place"""
        if self.encrypted:
            return NotImplemented
        out = np.round(self.data, decimals=decimals)
        return TensorBase(out)

    def round_(self, decimals=0):
        """Round the elements of tensor in-place to a nearest decimal place"""
        if self.encrypted:
            return NotImplemented
        self.data = np.round(self.data, decimals=decimals)
        return self

    def repeat(self, reps):
        """Return a new tensor by repeating the values given by reps"""
        if self.encrypted:
            return NotImplemented
        out = np.tile(self.data, reps=reps)
        return TensorBase(out)

    def pow(self, exponent):
        """Return a new tensor by raising elements to the given exponent.
        If exponent is an array, each element of the tensor is raised positionally to the
        element of the exponent"""
        if self.encrypted:
            return NotImplemented
        out = np.power(self.data, exponent)
        return TensorBase(out)

    def pow_(self, exponent):
        """Raise elements to the given exponent in-place. If exponent is an array,
        each element of the tensor is raised positionally to the element of the exponent"""
        if self.encrypted:
            return NotImplemented
        self.data = np.power(self.data, exponent)
        return self

    def prod(self, axis=None):
        """Returns a new tensor with the product of (specified axis) all the elements"""
        if self.encrypted:
            return NotImplemented
        out = np.prod(self.data, axis=axis)
        return TensorBase(out)

    def random_(self, low, high=None, size=None):
        """Fill the tensor in-place with random integers from [low to high)"""
        if self.encrypted:
            return NotImplemented
        self.data = np.random.randint(low=low, high=high, size=size)
        return self

    def nonzero(self):
        """Returns a new tensor with the indices of non-zero elements"""
        if self.encrypted:
            return NotImplemented
        out = np.array(np.nonzero(self.data))
        return TensorBase(out)

    def size(self):
        """Size of tensor"""
        if self.encrypted:
            return NotImplemented
        else:
            return self.data.size

    def cumprod(self, dim=0):
        """Returns the cumulative product of elements in the dimension dim."""
        if self.encrypted:
            return NotImplemented
        return syft.math.cumprod(self, dim)

    def cumprod_(self, dim=0):
        """calculate in-place the cumulative product of elements in the dimension dim."""
        if self.encrypted:
            return NotImplemented
        self.data = syft.math.cumprod(self, dim).data
        return self

    def split(self, split_size, dim=0):
        """Returns tuple of tensors of equally sized tensor/chunks (if possible)"""
        if self.encrypted:
            return NotImplemented
        splits = np.array_split(self.data, split_size, axis=0)
        tensors = list()
        for s in splits:
            tensors.append(TensorBase(s))
        tensors_tuple = tuple(tensors)
        return tensors_tuple

    def squeeze(self, axis=None):
        """Returns a new tensor with all the single-dimensional entries removed"""
        if self.encrypted:
            return NotImplemented
        out = np.squeeze(self.data, axis=axis)
        return TensorBase(out)

    def expand_as(self, tensor):
        """Returns a new tensor with the expanded size as of the specified (input) tensor"""
        if self.encrypted:
            return NotImplemented
        shape = tensor.data.shape
        neg_shapes = np.where(shape == -1)[0]
        if len(neg_shapes) > 1:
            shape[neg_shapes] = self.data.shape[neg_shapes]
        out = np.broadcast_to(self.data, shape)
        return TensorBase(out)

    def mean(self, dim=None, keepdim=False):
        """Return the mean of the tensor elements"""
        if self.encrypted:
            return NotImplemented
        out = np.mean(self.data, axis=dim, keepdims=keepdim)
        return TensorBase(out)

    def neg(self):
        """Returns negative of the elements of tensor"""
        if self.encrypted:
            return NotImplemented
        out = -1 * np.array(self.data)
        return TensorBase(out)

    def neg_(self):
        """Returns negative of the elements of tensor inplace"""
        if self.encrypted:
            return NotImplemented
        self.data = -1 * np.array(self.data)
        return self

    def normal(self, mu, sigma):
        """Returns a Tensor of random numbers drawn from separate
        normal distributions who’s mean and standard deviation are given."""
        if self.encrypted:
            return NotImplemented
        out = np.random.normal(mu, sigma, self.data.shape)
        return TensorBase(out)

    def normal_(self, mu, sigma):
        """Returns a Tensor of random numbers in-place drawn from separate
        normal distributions who’s mean and standard deviation are given."""
        if self.encrypted:
            return NotImplemented
        self.data = np.random.normal(mu, sigma, self.data.shape)
        return self

    def ne(self, tensor):
        """Checks element-wise equality with the given tensor and returns
        a boolean result with same dimension as the input matrix"""
        if self.encrypted:
            return NotImplemented
        else:
            if tensor.shape() == self.shape():

                tensor2 = np.array([1 if x else 0 for x in np.equal(
                    tensor.data.flatten(), self.data.flatten()).tolist()])
                result = tensor2.reshape(self.data.shape)
                return TensorBase(result)
            else:
                raise ValueError('inconsistent dimensions {} and {}'.format(
                    self.shape(), tensor.shape()))

    def ne_(self, tensor):
        """
         Checks in place element wise equality and updates the data matrix to the equality matrix
        """
        if self.encrypted:
            return NotImplemented
        else:
            value = self.ne(tensor)
            self.data = value.data

    def median(self, axis=1, keepdims=False):
        """Returns median of tensor as per specified axis. By default median is calculated along rows.
        axis=None can be used get median of whole tensor."""
        if self.encrypted:
            return NotImplemented
        out = np.median(np.array(self.data), axis=axis, keepdims=keepdims)
        return TensorBase(out)

    def mode(self, axis=1):
        """Returns mode of tensor as per specified axis. By default mode is calculated along rows.
        To get mode of whole tensor, specify axis=None"""
        if self.encrypted:
            return NotImplemented
        out = scipy.stats.mode(np.array(self.data), axis=axis)
        return TensorBase(out)

    def inverse(self):
        """Returns inverse of a square matrix"""
        if self.encrypted:
            return NotImplemented
        inv = np.linalg.inv(np.matrix(np.array(self.data)))
        return TensorBase(inv)

    def min(self, axis=1, keepdims=False):
        """Returns minimum value in tensor along rows by default
        but if axis=None it will return minimum value in tensor"""
        if self.encrypted:
            return NotImplemented
        min = np.matrix(np.array(self.data)).min(axis=axis, keepdims=keepdims)
        return TensorBase(min)

    def histc(self, bins=10, min=0, max=0):
        """Computes the histogram of a tensor and Returns it"""
        if self.encrypted:
            return NotImplemented
        hist, edges = np.histogram(
            np.array(self.data), bins=bins, range=(min, max))
        return TensorBase(hist)

    def scatter_(self, dim, index, src):
        """
        Writes all values from the Tensor ``src`` into ``self`` at the indices specified in the ``index`` Tensor.
        The indices are specified with respect to the given dimension, ``dim``, in the manner described in gather().
        :param dim: The axis along which to index
        :param index: The indices of elements to scatter
        :param src: The source element(s) to scatter
        :return: self
        """
        index = _ensure_tensorbase(index)
        if self.encrypted or index.encrypted:
            return NotImplemented
        if index.data.dtype != np.dtype('int_'):
            raise TypeError("The values of index must be integers")
        if self.data.ndim != index.data.ndim:
            raise ValueError(
                "Index should have the same number of dimensions as output")
        if dim >= self.data.ndim or dim < -self.data.ndim:
            raise IndexError("dim is out of range")
        if dim < 0:
            # Not sure why scatter should accept dim < 0, but that is the behavior in PyTorch's scatter
            dim = self.data.ndim + dim
        idx_xsection_shape = index.data.shape[:dim] + \
            index.data.shape[dim + 1:]
        self_xsection_shape = self.data.shape[:dim] + self.data.shape[dim + 1:]
        if idx_xsection_shape != self_xsection_shape:
            raise ValueError("Except for dimension " + str(dim) +
                             ", all dimensions of index and output should be the same size")
        if (index.data >= self.data.shape[dim]).any() or (index.data < 0).any():
            raise IndexError(
                "The values of index must be between 0 and (self.data.shape[dim] -1)")

        def make_slice(arr, dim, i):
            slc = [slice(None)] * arr.ndim
            slc[dim] = i
            return slc

        # We use index and dim parameters to create idx
        # idx is in a form that can be used as a NumPy advanced index for scattering of src param. in self.data
        idx = [[*np.indices(idx_xsection_shape).reshape(index.data.ndim - 1, -1),
                index.data[make_slice(index.data, dim, i)].reshape(1, -1)[0]] for i in range(index.data.shape[dim])]
        idx = list(np.concatenate(idx, axis=1))
        idx.insert(dim, idx.pop())

        if not np.isscalar(src):
            src = _ensure_tensorbase(src)
            if index.data.shape[dim] > src.data.shape[dim]:
                raise IndexError("Dimension " + str(dim) +
                                 "of index can not be bigger than that of src ")
            src_shape = src.data.shape[:dim] + src.data.shape[dim + 1:]
            if idx_xsection_shape != src_shape:
                raise ValueError("Except for dimension " +
                                 str(dim) + ", all dimensions of index and src should be the same size")
            # src_idx is a NumPy advanced index for indexing of elements in the src
            src_idx = list(idx)
            src_idx.pop(dim)
            src_idx.insert(dim, np.repeat(
                np.arange(index.data.shape[dim]), np.prod(idx_xsection_shape)))
            self.data[idx] = src.data[src_idx]

        else:
            self.data[idx] = src

        return self

    def gather(self, dim, index):
        """
        Gathers values along an axis specified by ``dim``.
        For a 3-D tensor the output is specified by:
            out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
            out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
            out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
        :param dim: The axis along which to index
        :param index: A tensor of indices of elements to gather
        :return: tensor of gathered values
        """
        index = _ensure_tensorbase(index)
        if self.encrypted or index.encrypted:
            return NotImplemented
        idx_xsection_shape = index.data.shape[:dim] + \
            index.data.shape[dim + 1:]
        self_xsection_shape = self.data.shape[:dim] + self.data.shape[dim + 1:]
        if idx_xsection_shape != self_xsection_shape:
            raise ValueError("Except for dimension " + str(dim) +
                             ", all dimensions of index and self should be the same size")
        if index.data.dtype != np.dtype('int_'):
            raise TypeError("The values of index must be integers")
        data_swaped = np.swapaxes(self.data, 0, dim)
        index_swaped = np.swapaxes(index, 0, dim)
        gathered = np.choose(index_swaped, data_swaped)
        return TensorBase(np.swapaxes(gathered, 0, dim))

    def serialize(self):
        return pickle.dumps(self)

    def deserialize(b):
        return pickle.loads(b)

    def remainder(self, divisor):
        """
        Computes the element-wise remainder of division.
        The divisor and dividend may contain both for integer and floating point numbers.
        The remainder has the same sign as the divisor.
        When ``divisor`` is a Tensor, the shapes of ``self`` and ``divisor`` must be broadcastable.
        :param divisor:  The divisor. This may be either a number or a tensor.
        :return: result tensor
        """
        if self.encrypted:
            return NotImplemented
        if not np.isscalar(divisor):
            divisor = _ensure_tensorbase(divisor)
        return TensorBase(np.remainder(self.data, divisor))

    def remainder_(self, divisor):
        """
        Computes the element-wise remainder of division.
        The divisor and dividend may contain both for integer and floating point numbers.
        The remainder has the same sign as the divisor.
        When ``divisor`` is a Tensor, the shapes of ``self`` and ``divisor`` must be broadcastable.
        :param divisor:  The divisor. This may be either a number or a tensor.
        :return: self
        """
        if self.encrypted:
            return NotImplemented
        if not np.isscalar(divisor):
            divisor = _ensure_tensorbase(divisor)
        self.data = np.remainder(self.data, divisor)
        return self

    def index_select(self, dim, index):
        """
        Returns a new Tensor which indexes the ``input`` Tensor along
        dimension ``dim`` using the entries in ``index``.

        :param dim: dimension in which to index
        :param index: 1D tensor containing the indices to index
        :return: Tensor of selected indices
        """
        index = _ensure_tensorbase(index)
        if self.encrypted or index.encrypted:
            return NotImplemented
        if index.data.ndim > 1:
            raise ValueError("Index is supposed to be 1D")
        return TensorBase(self.data.take(index, axis=dim))

    def mv(self, tensorvector):
        if self.encrypted:
            raise NotImplemented
        return mv(self, tensorvector)

    def masked_scatter_(self, mask, source):
        """
        Copies elements from ``source`` into this tensor at positions where the ``mask`` is true.
        The shape of ``mask`` must be broadcastable with the shape of the this tensor.
        The ``source`` should have at least as many elements as the number of ones in ``mask``.

        :param mask: The binary mask (non-zero is treated as true)
        :param source: The tensor to copy from
        :return:
        """
        mask = _ensure_tensorbase(mask)
        source = _ensure_tensorbase(source)
        if self.encrypted or mask.encrypted or source.encrypted:
            return NotImplemented
        mask_self_iter = np.nditer([mask.data, self.data])
        source_iter = np.nditer(source.data)
        out_flat = [s if m == 0 else source_iter.__next__().item()
                    for m, s in mask_self_iter]
        self.data = np.reshape(out_flat, self.data.shape)
        return self

    def masked_fill_(self, mask, value):
        """
        Fills elements of this ``tensor`` with value where ``mask`` is true.
        The shape of mask must be broadcastable with the shape of the underlying tensor.

        :param mask: The binary mask (non-zero is treated as true)
        :param value: value to fill
        :return:
        """
        mask = _ensure_tensorbase(mask)
        if self.encrypted or mask.encrypted:
            return NotImplemented
        if not np.isscalar(value):
            raise ValueError("'value' should be scalar")
        mask_broadcasted = np.broadcast_to(mask.data, self.data.shape)
        indices = np.where(mask_broadcasted)
        self.data[indices] = value
        return self

    def masked_select(self, mask):
        """
        See :func:`tensor.masked_select`
        """
        return masked_select(self, mask)

    def eq(self, t):
        """Returns a new Tensor having boolean True values where an element of the calling tensor is equal to the second Tensor, False otherwise.
        The second Tensor can be a number or a tensor whose shape is broadcastable with the calling Tensor."""
        if self.encrypted:
            return NotImplemented
        return TensorBase(np.equal(self.data, _ensure_tensorbase(t).data))

    def eq_(self, t):
        """Writes in-place, boolean True values where an element of the calling tensor is equal to the second Tensor, False otherwise.
        The second Tensor can be a number or a tensor whose shape is broadcastable with the calling Tensor."""
        if self.encrypted:
            return NotImplemented
        self.data = np.equal(self.data, _ensure_tensorbase(t).data)
        return self

    def mm(self, tensor2):
        """Performs a matrix multiplication of :attr:`tensor1` and :attr:`tensor2`.

        If :attr:`tensor1` is a `n x m` Tensor, :attr:`tensor2` is a `m x p` Tensor,
        output will be a `n x p` Tensor.

        Args:
            tensor1 (Tensor): First Tensor to be multiplied
            tensor2 (Tensor): Second Tensor to be multiplied"""

        return syft.mm(self, tensor2)


def mv(tensormat, tensorvector):
    """ matrix and vector multiplication """
    if tensormat.encrypted or tensorvector.encrypted:
        raise NotImplemented
    elif not len(tensorvector.data.shape) == 1:
        raise ValueError('Vector dimensions not correct {}'.format(
            tensorvector.data.shape))
    elif tensorvector.data.shape[0] != tensormat.data.shape[1]:
        raise ValueError('vector dimensions {} not  \
            compatible with matrix {} '.format(tensorvector.data.shape, tensormat.data.shape))
    else:
        return TensorBase(np.matmul(tensormat.data, tensorvector.data))


def masked_select(tensor, mask):
    """
    Returns a new 1D Tensor which indexes the ``input`` Tensor according to the binary mask ``mask``.
    The shapes of the ``mask`` tensor and the ``input`` tensor don’t need to match, but they must be broadcastable.

    :param tensor: Input tensor
    :param mask: The binary mask (non-zero is treated as true)
    :return: 1D output tensor
    """
    mask = _ensure_tensorbase(mask)
    tensor = _ensure_tensorbase(tensor)
    if tensor.encrypted or mask.encrypted:
        raise NotImplemented
    mask_broadcasted, data_broadcasted = np.broadcast_arrays(
        mask.data, tensor.data)
    indices = np.where(mask_broadcasted)
    return TensorBase(data_broadcasted[indices])
