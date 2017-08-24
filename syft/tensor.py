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

        tensor = _ensure_tensorbase(tensor)
        return TensorBase(self.data / tensor.data)

    def __itruediv__(self, tensor):
        """Performs in place element-wise subtraction between two tensors"""
        if self.encrypted:
            return NotImplemented

        tensor = _ensure_tensorbase(tensor)
        self.data /= tensor.data
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
        return np.ceil(self.data)

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
