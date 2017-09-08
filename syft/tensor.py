# -*- coding: utf-8 -*-
import numpy as np
import syft

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

    def __init__(self, arr_like, encrypted=False):
        self.data = _ensure_ndarray(arr_like)
        self.encrypted = encrypted

    def encrypt(self, pubkey):
        """Encrypts the Tensor using a Public Key"""
        if self.encrypted:
            return NotImplemented
        else:
            return pubkey.encrypt(self)

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
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

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
