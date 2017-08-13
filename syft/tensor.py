import numpy as np

def _ensure_ndarray(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    return arr

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

    def abs(self):
        """Returns absolute value of tensor as a new tensor"""
        if self.encrypted:
            return NotImplemented
        return np.absolute(self.data)
    
    def abs_(self):
        """Replaces tensor values with its absolute value"""
        if self.encrypted:
            return NotImplemented
        self.data=np.absolute(self.data)
        return self.data

    def shape(self):
        """Returns a tuple of input array dimensions."""
        if self.encrypted:
            return NotImplemented

        return self.data.shape

    def sum(self, dim=None):
        """Returns the sum of all elements in the input array."""
        if self.encrypted:
            return NotImplemented

        if dim is None:
            return self.data.sum()
        else:
            return self.data.sum(axis=dim)
     
     def addmm(self,tensor2,mat,beta=1,alpha=1):
        """Performs ((Mat*Beta)+((Tensor1.Tensor2)*Alpha)) and  returns the result as a Tensor
            Tensor1.Tensor2 is performed as Matrix product of two array The behavior depends on the arguments in the following way.
            *If both tensors are 1-dimensional, their dot product is returned.
            *If both arguments are 2-D they are multiplied like conventional matrices.
            *If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
            *If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
            *If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
            """
        if self.encrypted or tensor2.encrypted or mat.encrypted:
            return NotImplemented
        else:
            return TensorBase(np.array((mat*beta)+((np.matmul(self.data,tensor2.data))*alpha)))

    def addmm_(self,tensor2,mat,beta=1,alpha=1):
        """Performs ((Mat*Beta)+((Tensor1.Tensor2)*Alpha)) and updates Tensor1 with result and reurns it
            Tensor1.Tensor2 is performed as Matrix product of two array The behavior depends on the arguments in the following way.
            *If both tensors are 1-dimensional, their dot product is returned.
            *If both arguments are 2-D they are multiplied like conventional matrices.
            *If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
            *If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
            *If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
            """
        if self.encrypted is True or tensor2.encrypted is True or mat.encrypted is True:
            return NotImplemented
        else:
            self.data=np.array((mat*beta)+((np.matmul(self.data,tensor2.data))*alpha))
            return self

