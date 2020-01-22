import numpy as np
import syft as sy
import torch as th
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.overload import overloaded
from syft.frameworks.torch.tensors.interpreters.numpy import NumpyTensor
from syft.frameworks.torch.tensors.interpreters.polynomial.polynomial import Polynomial


class PolynomialTensor(NumpyTensor):
    """NumpyTensor is a tensor which seeks to wrap the Numpy API with the PyTorch tensor API.
    This is useful because Numpy can offer a wide range of existing functionality ranging from
    large precision, custom scalar types, and polynomial arithmetic.
    """

    def __init__(
        self, polynomials=None, owner=None, id=None, tags=None, description=None, verbose=False
    ):
        """Initializes a NumpyTensor.

        Args:
            numpy_tensor (np.array): The numpy array which this tensor should wrap.
            owner (BaseWorker): An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id (str or int): An optional string or integer id of the LargePrecisionTensor.
            tags (list): list of tags for searching.
            description (str): a description of this tensor.
        """
        super().__init__(
            numpy_tensor=polynomials, id=id, owner=owner, tags=tags, description=description
        )
        self.verbose = verbose

    def __call__(self, **kwargs):
        results = list()
        for x in self.child.flatten():
            results.append(x(**kwargs))
        return th.tensor(results)

    @overloaded.method
    def mm(self, _self, other):

        left, mid, right = _self.shape[0], _self.shape[1], other.shape[1]

        _self = np.expand_dims(_self, axis=2).repeat(repeats=right, axis=2)  # .expand(7,5,4)
        other = np.expand_dims(other, axis=0).repeat(repeats=left, axis=0)

        raw_result = _self * other

        result = raw_result.transpose(1, 0, 2)

        result = result.reshape(mid, left * right)

        result = mat_2d_sum0(result)

        result = result.reshape(left, right)

        return result

    @overloaded.method
    def sum(self, _self, dim):
        """Because of the way polynomial addition works, we can get a very significant performance increase
        (2-3 orders of magnitude) by taking into account the knoweldge that we're summing over the entire vector."""

        matrix = _self

        if len(matrix.shape) == 2:

            if dim == 0:
                return mat_2d_sum0(matrix)
            elif dim == 1:
                return mat_2d_sum1(matrix)
            else:
                print(
                    "Dim " + str(dim) + " out of range for matrix with shape " + str(matrix.shape)
                )
        elif len(matrix.shape) == 1:

            if dim == 0:
                result = vsum(matrix)
                return np.array(result)
            else:
                print(
                    "Dim " + str(dim) + " out of range for matrix with shape " + str(matrix.shape)
                )
        else:

            result = _self.sum(dim)
            result = np.array(result)
            return result


def vsum(vector):
    additive_terms = list()
    additive_constant = 0

    for p in vector.flatten():
        for at in p.additive_terms:
            additive_terms.append(at)
        additive_constant += p.additive_constant

    return Polynomial(
        terms=additive_terms, constant=additive_constant, minimum_factor=p.minimum_factor
    ).reduce()


# def mat_2d_sum0(matrix):
#
#     inputs = list()
#     for i in range(matrix.shape[1]):
#         inputs.append(matrix[:, i])
#
#     results = sy.pool().map(vsum, inputs)
#
#     return np.array(results)


def mat_2d_sum0(matrix):
    results = list()
    for i in range(matrix.shape[1]):
        results.append(vsum(matrix[:, i]))

    return np.array(results)


def mat_2d_sum1(matrix):
    return mat_2d_sum0(matrix.transpose())


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(PolynomialTensor)
